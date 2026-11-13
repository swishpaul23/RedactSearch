variable "region" {
  description = "AWS region for every resource (US West, Oregon)."
  type        = string
  default     = "us-west-2"
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

variable "api_image_tag" {
  description = "Tag of the web API image (Dockerfile.api) in ECR. Leave empty until that image is pushed."
  type        = string
  default     = ""
}

variable "anthropic_api_key" {
  description = "Key the API uses for answers. Set it with the TF_VAR_anthropic_api_key environment variable, not in a file. Terraform state holds it in plain text, which is why state stays local and gitignored."
  type        = string
  default     = ""
  sensitive   = true
}

variable "api_memory_mb" {
  description = "API Lambda memory. CPU scales with memory; model loading and BM25 run on CPU."
  type        = number
  default     = 3008
}

variable "api_timeout_s" {
  description = "API Lambda timeout. Vercel stops waiting for a proxied request after 120 seconds."
  type        = number
  default     = 120
}

variable "api_ask_per_minute" {
  description = "Questions per minute per visitor (per warm instance)."
  type        = number
  default     = 10
}
