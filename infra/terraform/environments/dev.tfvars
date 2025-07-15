environment             = "dev"
location                = "eastus2"
zones                   = ["1"]
workload_vm_size        = "Standard_D4ds_v5"
workload_node_count     = { min = 1, max = 3 }
eventhub_capacity       = 1
transactions_partitions = 4
redis_sku               = { family = "C", sku_name = "Standard", capacity = 1 }
