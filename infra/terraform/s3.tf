# Two buckets on either side of the redaction boundary.
#   raw:      uploads land in incoming/. Encrypted with KMS (AWS-managed aws/s3 key, no monthly fee),
#             private, TLS only, and deleted after raw_retention_days.
#   redacted: only the Lambda writes here (redacted/). The index is built from this bucket alone.

resource "random_id" "suffix" {
  byte_length = 4 # bucket names are global; the suffix keeps them unique
}

locals {
  raw_prefix = "incoming/"
  out_prefix = "redacted/"
}

resource "aws_s3_bucket" "raw" {
  bucket        = "${var.project}-raw-${random_id.suffix.hex}"
  force_destroy = true # demo data only; lets `terraform destroy` remove a non-empty bucket
}

resource "aws_s3_bucket" "redacted" {
  bucket        = "${var.project}-redacted-${random_id.suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "raw" {
  bucket                  = aws_s3_bucket.raw.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "redacted" {
  bucket                  = aws_s3_bucket.redacted.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "raw" {
  bucket = aws_s3_bucket.raw.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_ownership_controls" "redacted" {
  bucket = aws_s3_bucket.redacted.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms" # no key id: the AWS-managed aws/s3 key
    }
    bucket_key_enabled = true # fewer KMS requests, lower cost
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "redacted" {
  bucket = aws_s3_bucket.redacted.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id

  rule {
    id     = "expire-raw-transcripts"
    status = "Enabled"
    filter {}
    expiration {
      days = var.raw_retention_days
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

data "aws_iam_policy_document" "tls_only_raw" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.raw.arn, "${aws_s3_bucket.raw.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

data "aws_iam_policy_document" "tls_only_redacted" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.redacted.arn, "${aws_s3_bucket.redacted.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "raw" {
  bucket     = aws_s3_bucket.raw.id
  policy     = data.aws_iam_policy_document.tls_only_raw.json
  depends_on = [aws_s3_bucket_public_access_block.raw]
}

resource "aws_s3_bucket_policy" "redacted" {
  bucket     = aws_s3_bucket.redacted.id
  policy     = data.aws_iam_policy_document.tls_only_redacted.json
  depends_on = [aws_s3_bucket_public_access_block.redacted]
}
