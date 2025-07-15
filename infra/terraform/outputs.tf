output "resource_group" {
  description = "Resource group holding every Tessera resource."
  value       = azurerm_resource_group.main.name
}

output "aks_name" {
  description = "AKS cluster name, for `az aks get-credentials`."
  value       = azurerm_kubernetes_cluster.main.name
}

output "acr_login_server" {
  description = "Container registry the CI pipeline pushes to."
  value       = azurerm_container_registry.main.login_server
}

output "key_vault_name" {
  description = "Key Vault holding runtime secrets (Helm value keyVault.name)."
  value       = azurerm_key_vault.main.name
}

output "lake_account" {
  description = "ADLS Gen2 account for archives, training sets, models and feature logs."
  value       = azurerm_storage_account.lake.name
}

output "workload_client_ids" {
  description = "Client IDs for the Helm values identity.<workload>.clientId."
  value       = { for k, v in azurerm_user_assigned_identity.workload : k => v.client_id }
}

output "tenant_id" {
  description = "Entra ID tenant (Helm value identity.tenantId)."
  value       = data.azurerm_client_config.current.tenant_id
}

output "kafka_bootstrap" {
  description = "Event Hubs Kafka endpoint."
  value       = "${azurerm_eventhub_namespace.main.name}.servicebus.windows.net:9093"
}
