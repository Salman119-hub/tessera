data "azurerm_client_config" "current" {}

resource "random_string" "suffix" {
  length  = 5
  upper   = false
  special = false
}

locals {
  name    = "tessera-${var.environment}"
  compact = "tessera${var.environment}${random_string.suffix.result}"
  is_prod = var.environment == "prod"
  tags    = merge({ project = "tessera", environment = var.environment, managed_by = "terraform" }, var.tags)

  # Kubernetes service accounts that exchange their tokens for Azure identities.
  workloads = {
    api    = "tessera-api"
    stream = "tessera-stream"
    batch  = "tessera-batch"
  }
}

resource "azurerm_resource_group" "main" {
  name     = "rg-${local.name}"
  location = var.location
  tags     = local.tags
}

# --------------------------------------------------------------------------- observability

resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = local.is_prod ? 90 : 30
  tags                = local.tags
}

# --------------------------------------------------------------------------- network

resource "azurerm_virtual_network" "main" {
  name                = "vnet-${local.name}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  address_space       = [var.vnet_cidr]
  tags                = local.tags
}

resource "azurerm_subnet" "aks" {
  name                 = "snet-aks"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = [cidrsubnet(var.vnet_cidr, 4, 0)]
  service_endpoints    = ["Microsoft.Storage", "Microsoft.KeyVault", "Microsoft.EventHub"]
}

# --------------------------------------------------------------------------- registry

resource "azurerm_container_registry" "main" {
  name                          = local.compact
  location                      = azurerm_resource_group.main.location
  resource_group_name           = azurerm_resource_group.main.name
  sku                           = local.is_prod ? "Premium" : "Standard"
  admin_enabled                 = false
  public_network_access_enabled = true
  tags                          = local.tags
}

# --------------------------------------------------------------------------- AKS

resource "azurerm_kubernetes_cluster" "main" {
  name                      = "aks-${local.name}"
  location                  = azurerm_resource_group.main.location
  resource_group_name       = azurerm_resource_group.main.name
  dns_prefix                = local.name
  kubernetes_version        = var.kubernetes_version
  sku_tier                  = local.is_prod ? "Standard" : "Free"
  automatic_upgrade_channel = "patch"
  oidc_issuer_enabled       = true
  workload_identity_enabled = true
  local_account_disabled    = true

  default_node_pool {
    name                         = "system"
    vm_size                      = var.system_vm_size
    vnet_subnet_id               = azurerm_subnet.aks.id
    auto_scaling_enabled         = true
    min_count                    = 2
    max_count                    = 4
    zones                        = var.zones
    only_critical_addons_enabled = true
    os_disk_type                 = "Ephemeral"

    upgrade_settings {
      max_surge = "33%"
    }
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin      = "azure"
    network_plugin_mode = "overlay"
    network_policy      = "azure"
    load_balancer_sku   = "standard"
  }

  azure_active_directory_role_based_access_control {
    azure_rbac_enabled = true
    tenant_id          = data.azurerm_client_config.current.tenant_id
  }

  key_vault_secrets_provider {
    secret_rotation_enabled = true
  }

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  tags = local.tags
}

resource "azurerm_kubernetes_cluster_node_pool" "workload" {
  name                  = "workload"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = var.workload_vm_size
  vnet_subnet_id        = azurerm_subnet.aks.id
  auto_scaling_enabled  = true
  min_count             = var.workload_node_count.min
  max_count             = var.workload_node_count.max
  zones                 = var.zones
  os_disk_type          = "Ephemeral"
  node_labels           = { "tessera.io/pool" = "workload" }
  tags                  = local.tags

  upgrade_settings {
    max_surge = "33%"
  }
}

resource "azurerm_role_assignment" "aks_acr_pull" {
  scope                            = azurerm_container_registry.main.id
  role_definition_name             = "AcrPull"
  principal_id                     = azurerm_kubernetes_cluster.main.kubelet_identity[0].object_id
  skip_service_principal_aad_check = true
}

# --------------------------------------------------------------------------- Event Hubs (Kafka API)

resource "azurerm_eventhub_namespace" "main" {
  name                     = "evhns-${local.name}-${random_string.suffix.result}"
  location                 = azurerm_resource_group.main.location
  resource_group_name      = azurerm_resource_group.main.name
  sku                      = "Standard" # Standard and above expose the Kafka endpoint on :9093
  capacity                 = var.eventhub_capacity
  auto_inflate_enabled     = true
  maximum_throughput_units = var.eventhub_capacity * 4
  minimum_tls_version      = "1.2"
  tags                     = local.tags
}

resource "azurerm_eventhub" "transactions" {
  name              = "payments.transactions.v1"
  namespace_id      = azurerm_eventhub_namespace.main.id
  partition_count   = var.transactions_partitions
  message_retention = 7
}

resource "azurerm_eventhub" "feature_log" {
  name              = "tessera.feature-log.v1"
  namespace_id      = azurerm_eventhub_namespace.main.id
  partition_count   = 4
  message_retention = 3
}

resource "azurerm_eventhub_consumer_group" "stream" {
  name                = "tessera-stream"
  namespace_name      = azurerm_eventhub_namespace.main.name
  eventhub_name       = azurerm_eventhub.transactions.name
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_eventhub_authorization_rule" "stream_listen" {
  name                = "tessera-stream-listen"
  namespace_name      = azurerm_eventhub_namespace.main.name
  eventhub_name       = azurerm_eventhub.transactions.name
  resource_group_name = azurerm_resource_group.main.name
  listen              = true
  send                = false
  manage              = false
}

# --------------------------------------------------------------------------- online store

resource "azurerm_redis_cache" "online" {
  name                 = "redis-${local.name}-${random_string.suffix.result}"
  location             = azurerm_resource_group.main.location
  resource_group_name  = azurerm_resource_group.main.name
  family               = var.redis_sku.family
  sku_name             = var.redis_sku.sku_name
  capacity             = var.redis_sku.capacity
  non_ssl_port_enabled = false
  minimum_tls_version  = "1.2"
  zones                = var.redis_sku.sku_name == "Premium" ? var.zones : null

  redis_configuration {
    # Feature state must never be evicted silently; a full cache should page.
    maxmemory_policy = "noeviction"
  }

  tags = local.tags
}

# --------------------------------------------------------------------------- lake (offline store)

resource "azurerm_storage_account" "lake" {
  name                            = local.compact
  location                        = azurerm_resource_group.main.location
  resource_group_name             = azurerm_resource_group.main.name
  account_tier                    = "Standard"
  account_kind                    = "StorageV2"
  account_replication_type        = local.is_prod ? "ZRS" : "LRS"
  is_hns_enabled                  = true
  min_tls_version                 = "TLS1_2"
  shared_access_key_enabled       = false
  allow_nested_items_to_be_public = false

  blob_properties {
    delete_retention_policy {
      days = 14
    }
    container_delete_retention_policy {
      days = 14
    }
  }

  tags = local.tags
}

resource "azurerm_storage_container" "this" {
  for_each              = toset(["lake", "models"])
  name                  = each.key
  storage_account_id    = azurerm_storage_account.lake.id
  container_access_type = "private"
}

resource "azurerm_storage_management_policy" "lake" {
  storage_account_id = azurerm_storage_account.lake.id

  rule {
    name    = "feature-log-retention"
    enabled = true
    filters {
      blob_types   = ["blockBlob"]
      prefix_match = ["lake/feature-log/"]
    }
    actions {
      base_blob {
        tier_to_cool_after_days_since_modification_greater_than = 30
        delete_after_days_since_modification_greater_than       = 400
      }
    }
  }
}

# --------------------------------------------------------------------------- identities

resource "azurerm_user_assigned_identity" "workload" {
  for_each            = local.workloads
  name                = "id-${local.name}-${each.key}"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  tags                = local.tags
}

resource "azurerm_federated_identity_credential" "workload" {
  for_each            = local.workloads
  name                = "aks-${each.value}"
  resource_group_name = azurerm_resource_group.main.name
  parent_id           = azurerm_user_assigned_identity.workload[each.key].id
  audience            = ["api://AzureADTokenExchange"]
  issuer              = azurerm_kubernetes_cluster.main.oidc_issuer_url
  subject             = "system:serviceaccount:${var.k8s_namespace}:${each.value}"
}

resource "azurerm_role_assignment" "models_read" {
  scope                = "${azurerm_storage_account.lake.id}/blobServices/default/containers/models"
  role_definition_name = "Storage Blob Data Reader"
  principal_id         = azurerm_user_assigned_identity.workload["api"].principal_id
  depends_on           = [azurerm_storage_container.this]
}

resource "azurerm_role_assignment" "batch_lake" {
  scope                = azurerm_storage_account.lake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_user_assigned_identity.workload["batch"].principal_id
}

resource "azurerm_role_assignment" "stream_receive" {
  scope                = azurerm_eventhub.transactions.id
  role_definition_name = "Azure Event Hubs Data Receiver"
  principal_id         = azurerm_user_assigned_identity.workload["stream"].principal_id
}

# --------------------------------------------------------------------------- secrets

resource "azurerm_key_vault" "main" {
  name                       = "kv-${local.name}-${random_string.suffix.result}"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  enable_rbac_authorization  = true
  purge_protection_enabled   = local.is_prod
  soft_delete_retention_days = 7
  tags                       = local.tags
}

resource "azurerm_role_assignment" "deployer_secrets" {
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_role_assignment" "workload_secrets" {
  for_each             = local.workloads
  scope                = azurerm_key_vault.main.id
  role_definition_name = "Key Vault Secrets User"
  principal_id         = azurerm_user_assigned_identity.workload[each.key].principal_id
}

resource "random_password" "api_key" {
  length  = 40
  special = false
}

locals {
  secrets = {
    "online-store-url"            = "rediss://:${azurerm_redis_cache.online.primary_access_key}@${azurerm_redis_cache.online.hostname}:${azurerm_redis_cache.online.ssl_port}/0"
    "eventhubs-connection-string" = azurerm_eventhub_authorization_rule.stream_listen.primary_connection_string
    "eventhubs-bootstrap"         = "${azurerm_eventhub_namespace.main.name}.servicebus.windows.net:9093"
    "api-keys"                    = jsonencode([random_password.api_key.result])
  }
}

resource "azurerm_key_vault_secret" "this" {
  for_each     = local.secrets
  name         = each.key
  value        = each.value
  key_vault_id = azurerm_key_vault.main.id
  content_type = "text/plain"
  depends_on   = [azurerm_role_assignment.deployer_secrets]
}
