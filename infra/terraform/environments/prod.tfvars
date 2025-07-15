environment             = "prod"
location                = "eastus2"
zones                   = ["1", "2", "3"]
workload_vm_size        = "Standard_D8ds_v5"
workload_node_count     = { min = 3, max = 12 }
eventhub_capacity       = 4
transactions_partitions = 24
redis_sku               = { family = "P", sku_name = "Premium", capacity = 2 }
