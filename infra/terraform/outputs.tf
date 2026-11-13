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
