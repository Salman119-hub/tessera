variable "subscription_id" {
  description = "Azure subscription to deploy into."
  type        = string
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be dev, staging or prod."
  }
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "eastus2"
}

variable "zones" {
  description = "Availability zones for zonal resources."
  type        = list(string)
  default     = ["1", "2", "3"]
}

variable "kubernetes_version" {
  description = "AKS control plane version."
  type        = string
  default     = "1.30"
}

variable "vnet_cidr" {
  description = "Address space for the virtual network."
  type        = string
  default     = "10.40.0.0/16"
}

variable "system_vm_size" {
  description = "VM size for the AKS system node pool."
  type        = string
  default     = "Standard_D4ds_v5"
}

variable "workload_vm_size" {
  description = "VM size for the workload node pool (API and stream processor)."
  type        = string
  default     = "Standard_D8ds_v5"
}

variable "workload_node_count" {
  description = "Min and max node count for the workload pool."
  type        = object({ min = number, max = number })
  default     = { min = 2, max = 10 }
}

variable "eventhub_capacity" {
  description = "Event Hubs throughput units (auto-inflate raises this up to 4x)."
  type        = number
  default     = 2
}

variable "transactions_partitions" {
  description = "Partitions for the transactions hub; upper bound on stream replicas."
  type        = number
  default     = 12
}

variable "redis_sku" {
  description = "Azure Cache for Redis SKU: family, sku_name and capacity."
  type        = object({ family = string, sku_name = string, capacity = number })
  default     = { family = "C", sku_name = "Standard", capacity = 1 }
}

variable "k8s_namespace" {
  description = "Namespace the Helm release is installed into."
  type        = string
  default     = "tessera"
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
