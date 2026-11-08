# Container image registry, the redaction Lambda, its permissions, and the S3 trigger.
# The Lambda (and everything that needs it) is created only once image_tag is set,
# because Lambda cannot be created before its image exists in ECR.

data "aws_caller_identity" "current" {}

locals {
  deploy_lambda = var.image_tag != ""
  function_name = "${var.project}-redact"
}

resource "aws_ecr_repository" "lambda" {
  name                 = "${var.project}-redact"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "lambda" {
  repository = aws_ecr_repository.lambda.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 3 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 3
      }
      action = { type = "expire" }
    }]
  })
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least privilege: read raw uploads, write redacted output, nothing else.
data "aws_iam_policy_document" "lambda_s3" {
  statement {
    sid       = "ReadRawUploads"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.raw.arn}/${local.raw_prefix}*"]
  }
  statement {
    sid       = "WriteRedacted"
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.redacted.arn}/${local.out_prefix}*"]
  }
  statement {
    sid       = "DecryptRawViaS3"
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["s3.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "lambda_s3" {
  name   = "s3-redaction-boundary"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_s3.json
}

resource "aws_cloudwatch_log_group" "lambda" {
  count             = local.deploy_lambda ? 1 : 0
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 14
}

resource "aws_lambda_function" "redact" {
  count         = local.deploy_lambda ? 1 : 0
  function_name = local.function_name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.lambda.repository_url}:${var.image_tag}"
  architectures = ["x86_64"]
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_s

  environment {
    variables = {
      REDACTED_BUCKET = aws_s3_bucket.redacted.id
      RAW_PREFIX      = local.raw_prefix
    }
  }

  depends_on = [aws_cloudwatch_log_group.lambda, aws_iam_role_policy_attachment.lambda_logs]
}

resource "aws_lambda_permission" "s3_invoke" {
  count          = local.deploy_lambda ? 1 : 0
  statement_id   = "AllowRawBucketInvoke"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.redact[0].function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.raw.arn
  source_account = data.aws_caller_identity.current.account_id
}

resource "aws_s3_bucket_notification" "raw_upload" {
  count  = local.deploy_lambda ? 1 : 0
  bucket = aws_s3_bucket.raw.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.redact[0].arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = local.raw_prefix
    filter_suffix       = ".jsonl"
  }

  depends_on = [aws_lambda_permission.s3_invoke]
}
