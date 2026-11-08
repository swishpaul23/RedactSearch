variable "region" {
  description = "AWS region for every resource. Canada (Central) keeps the transcripts in Canada."
  type        = string
  default     = "ca-central-1"
}

variable "project" {
  description = "Name prefix for resources."
  type        = string
  default     = "callsearch"
}

variable "alert_email" {
  description = "Email that receives the monthly budget alerts."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget. An email is sent at 80% of actual spend and at 100% of forecast."
  type        = number
  default     = 5
}

variable "image_tag" {
  description = "Tag of the Lambda image in ECR. Leave empty on the first apply (creates buckets and ECR only), then push an image and apply again with its tag."
  type        = string
  default     = ""
}

variable "raw_retention_days" {
  description = "Raw transcripts are deleted this many days after upload."
  type        = number
  default     = 7
}

variable "lambda_memory_mb" {
  description = "Lambda memory. CPU scales with memory; DistilBERT runs on CPU."
  type        = number
  default     = 3008
}

variable "lambda_timeout_s" {
  description = "Lambda timeout per batch, in seconds (max 900)."
  type        = number
  default     = 600
}
