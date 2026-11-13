output "region" {
  value = var.region
}

output "raw_bucket" {
  value = aws_s3_bucket.raw.id
}

output "redacted_bucket" {
  value = aws_s3_bucket.redacted.id
}

output "ecr_repository_url" {
  value = aws_ecr_repository.lambda.repository_url
}

output "lambda_function_name" {
  value = local.deploy_lambda ? aws_lambda_function.redact[0].function_name : "(not deployed: set image_tag)"
}

output "api_ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "api_url" {
  description = "Function URL of the web API; put its host in web/vercel.json."
  value       = local.deploy_api ? aws_lambda_function_url.api[0].function_url : "(not deployed: set api_image_tag)"
}
