# The web API: Dockerfile.api on Lambda behind a public function URL, which Vercel proxies
# /api/* to. Like the redaction Lambda, the function exists only once api_image_tag is set,
# because Lambda cannot be created before its image is in ECR. Its role can write logs and
# nothing else: the API reads only what is baked into the image.

locals {
  deploy_api        = var.api_image_tag != ""
  api_function_name = "${var.project}-api"
}

resource "aws_ecr_repository" "api" {
  name                 = local.api_function_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 2 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 2
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_iam_role" "api" {
  name               = "${local.api_function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "api_logs" {
  role       = aws_iam_role.api.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_cloudwatch_log_group" "api" {
  count             = local.deploy_api ? 1 : 0
  name              = "/aws/lambda/${local.api_function_name}"
  retention_in_days = 14
}

resource "aws_lambda_function" "api" {
  count         = local.deploy_api ? 1 : 0
  function_name = local.api_function_name
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
  architectures = ["x86_64"]
  memory_size   = var.api_memory_mb
  timeout       = var.api_timeout_s

  environment {
    variables = {
      ANTHROPIC_API_KEY         = var.anthropic_api_key
      CALLSEARCH_ASK_PER_MINUTE = tostring(var.api_ask_per_minute)
      CALLSEARCH_UPLOADS        = "off"
    }
  }

  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy_attachment.api_logs]
}

resource "aws_lambda_function_url" "api" {
  count              = local.deploy_api ? 1 : 0
  function_name      = aws_lambda_function.api[0].function_name
  authorization_type = "NONE"
}

# A public function URL needs both permissions (required for URLs created since October 2025).
resource "aws_lambda_permission" "api_url" {
  count                  = local.deploy_api ? 1 : 0
  statement_id           = "AllowPublicFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.api[0].function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "api_invoke" {
  count                    = local.deploy_api ? 1 : 0
  statement_id             = "AllowInvokeViaFunctionUrl"
  action                   = "lambda:InvokeFunction"
  function_name            = aws_lambda_function.api[0].function_name
  principal                = "*"
  invoked_via_function_url = true
}
