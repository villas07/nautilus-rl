#!/usr/bin/env python3
"""
Model Download/Upload Script

Handles model transfer between local/VPS and cloud storage.

Usage:
    python download_models.py --download              # Download all models
    python download_models.py --upload-models         # Upload trained models
    python download_models.py --sync                  # Bi-directional sync
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import argparse
import shutil

import structlog

logger = structlog.get_logger()

# Try to import cloud storage libraries
try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    from google.cloud import storage as gcs
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class ModelManager:
    """
    Manages model storage and retrieval.

    Supports:
    - Local filesystem
    - AWS S3
    - Google Cloud Storage
    - Direct SFTP to VPS
    """

    def __init__(
        self,
        local_dir: str = "/app/models",
        remote_uri: Optional[str] = None,
    ):
        """
        Initialize model manager.

        Args:
            local_dir: Local model directory.
            remote_uri: Remote storage URI (s3://, gs://, or sftp://).
        """
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.remote_uri = remote_uri or os.getenv("MODEL_STORAGE_URI")
        self.storage_type = self._detect_storage_type()

    def _detect_storage_type(self) -> str:
        """Detect storage type from URI."""
        if not self.remote_uri:
            return "local"
        if self.remote_uri.startswith("s3://"):
            return "s3"
        if self.remote_uri.startswith("gs://"):
            return "gcs"
        if self.remote_uri.startswith("sftp://"):
            return "sftp"
        return "local"

    def list_local_models(self) -> List[str]:
        """List models in local directory."""
        models = []
        for path in self.local_dir.glob("**/best_model.zip"):
            agent_id = path.parent.parent.name
            models.append(agent_id)
        for path in self.local_dir.glob("**/*_final.zip"):
            agent_id = path.stem.replace("_final", "")
            models.append(agent_id)
        return sorted(set(models))

    def list_remote_models(self) -> List[str]:
        """List models in remote storage."""
        if self.storage_type == "s3":
            return self._list_s3_models()
        elif self.storage_type == "gcs":
            return self._list_gcs_models()
        elif self.storage_type == "sftp":
            return self._list_sftp_models()
        return []

    def _list_s3_models(self) -> List[str]:
        """List models in S3."""
        if not S3_AVAILABLE:
            logger.warning("boto3 not installed")
            return []

        bucket, prefix = self._parse_s3_uri()
        s3 = boto3.client("s3")

        models = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".zip"):
                    # Extract agent ID from path
                    parts = obj["Key"].split("/")
                    for part in parts:
                        if part.startswith("agent_"):
                            models.append(part)
                            break

        return sorted(set(models))

    def _list_gcs_models(self) -> List[str]:
        """List models in GCS."""
        if not GCS_AVAILABLE:
            logger.warning("google-cloud-storage not installed")
            return []

        bucket_name, prefix = self._parse_gcs_uri()
        client = gcs.Client()
        bucket = client.bucket(bucket_name)

        models = []
        for blob in bucket.list_blobs(prefix=prefix):
            if blob.name.endswith(".zip"):
                parts = blob.name.split("/")
                for part in parts:
                    if part.startswith("agent_"):
                        models.append(part)
                        break

        return sorted(set(models))

    def _list_sftp_models(self) -> List[str]:
        """List models via SFTP."""
        # Implementation would use paramiko
        logger.warning("SFTP listing not implemented")
        return []

    def download_model(
        self,
        agent_id: str,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Download a specific model.

        Args:
            agent_id: Agent identifier.
            overwrite: Overwrite existing local model.

        Returns:
            Path to downloaded model.
        """
        local_path = self.local_dir / agent_id / f"{agent_id}_final.zip"

        if local_path.exists() and not overwrite:
            logger.info(f"Model {agent_id} already exists locally")
            return local_path

        if self.storage_type == "s3":
            return self._download_from_s3(agent_id, local_path)
        elif self.storage_type == "gcs":
            return self._download_from_gcs(agent_id, local_path)
        elif self.storage_type == "sftp":
            return self._download_from_sftp(agent_id, local_path)

        return None

    def _download_from_s3(self, agent_id: str, local_path: Path) -> Optional[Path]:
        """Download from S3."""
        if not S3_AVAILABLE:
            return None

        bucket, prefix = self._parse_s3_uri()
        s3 = boto3.client("s3")

        remote_key = f"{prefix}/{agent_id}/{agent_id}_final.zip"

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, remote_key, str(local_path))
            logger.info(f"Downloaded {agent_id} from S3")
            return local_path
        except Exception as e:
            logger.error(f"S3 download failed for {agent_id}: {e}")
            return None

    def _download_from_gcs(self, agent_id: str, local_path: Path) -> Optional[Path]:
        """Download from GCS."""
        if not GCS_AVAILABLE:
            return None

        bucket_name, prefix = self._parse_gcs_uri()
        client = gcs.Client()
        bucket = client.bucket(bucket_name)

        remote_key = f"{prefix}/{agent_id}/{agent_id}_final.zip"
        blob = bucket.blob(remote_key)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
            logger.info(f"Downloaded {agent_id} from GCS")
            return local_path
        except Exception as e:
            logger.error(f"GCS download failed for {agent_id}: {e}")
            return None

    def _download_from_sftp(self, agent_id: str, local_path: Path) -> Optional[Path]:
        """Download via SFTP."""
        # Implementation would use paramiko
        logger.warning("SFTP download not implemented")
        return None

    def download_all(
        self,
        overwrite: bool = False,
        filter_prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Download all models from remote storage.

        Args:
            overwrite: Overwrite existing models.
            filter_prefix: Only download models matching prefix.

        Returns:
            List of downloaded agent IDs.
        """
        remote_models = self.list_remote_models()

        if filter_prefix:
            remote_models = [m for m in remote_models if m.startswith(filter_prefix)]

        downloaded = []
        for agent_id in remote_models:
            if self.download_model(agent_id, overwrite):
                downloaded.append(agent_id)

        logger.info(f"Downloaded {len(downloaded)} models")
        return downloaded

    def upload_model(
        self,
        agent_id: str,
        overwrite: bool = False,
    ) -> bool:
        """
        Upload a model to remote storage.

        Args:
            agent_id: Agent identifier.
            overwrite: Overwrite existing remote model.

        Returns:
            True if successful.
        """
        # Find local model file
        local_path = self.local_dir / agent_id / f"{agent_id}_final.zip"
        if not local_path.exists():
            local_path = self.local_dir / agent_id / "best" / "best_model.zip"

        if not local_path.exists():
            logger.warning(f"Model not found: {agent_id}")
            return False

        if self.storage_type == "s3":
            return self._upload_to_s3(agent_id, local_path)
        elif self.storage_type == "gcs":
            return self._upload_to_gcs(agent_id, local_path)
        elif self.storage_type == "sftp":
            return self._upload_to_sftp(agent_id, local_path)

        return False

    def _upload_to_s3(self, agent_id: str, local_path: Path) -> bool:
        """Upload to S3."""
        if not S3_AVAILABLE:
            return False

        bucket, prefix = self._parse_s3_uri()
        s3 = boto3.client("s3")

        remote_key = f"{prefix}/{agent_id}/{agent_id}_final.zip"

        try:
            s3.upload_file(str(local_path), bucket, remote_key)
            logger.info(f"Uploaded {agent_id} to S3")
            return True
        except Exception as e:
            logger.error(f"S3 upload failed for {agent_id}: {e}")
            return False

    def _upload_to_gcs(self, agent_id: str, local_path: Path) -> bool:
        """Upload to GCS."""
        if not GCS_AVAILABLE:
            return False

        bucket_name, prefix = self._parse_gcs_uri()
        client = gcs.Client()
        bucket = client.bucket(bucket_name)

        remote_key = f"{prefix}/{agent_id}/{agent_id}_final.zip"
        blob = bucket.blob(remote_key)

        try:
            blob.upload_from_filename(str(local_path))
            logger.info(f"Uploaded {agent_id} to GCS")
            return True
        except Exception as e:
            logger.error(f"GCS upload failed for {agent_id}: {e}")
            return False

    def _upload_to_sftp(self, agent_id: str, local_path: Path) -> bool:
        """Upload via SFTP."""
        logger.warning("SFTP upload not implemented")
        return False

    def upload_all(self) -> List[str]:
        """Upload all local models."""
        local_models = self.list_local_models()

        uploaded = []
        for agent_id in local_models:
            if self.upload_model(agent_id):
                uploaded.append(agent_id)

        logger.info(f"Uploaded {len(uploaded)} models")
        return uploaded

    def sync(self) -> Dict[str, List[str]]:
        """
        Sync models between local and remote.

        Downloads missing remote models, uploads missing local models.
        """
        local_models = set(self.list_local_models())
        remote_models = set(self.list_remote_models())

        to_download = remote_models - local_models
        to_upload = local_models - remote_models

        downloaded = []
        for agent_id in to_download:
            if self.download_model(agent_id):
                downloaded.append(agent_id)

        uploaded = []
        for agent_id in to_upload:
            if self.upload_model(agent_id):
                uploaded.append(agent_id)

        return {
            "downloaded": downloaded,
            "uploaded": uploaded,
        }

    def _parse_s3_uri(self) -> tuple:
        """Parse S3 URI into bucket and prefix."""
        # s3://bucket/prefix
        path = self.remote_uri.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    def _parse_gcs_uri(self) -> tuple:
        """Parse GCS URI into bucket and prefix."""
        # gs://bucket/prefix
        path = self.remote_uri.replace("gs://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage model storage")

    parser.add_argument("--local-dir", default="/app/models", help="Local directory")
    parser.add_argument("--remote-uri", help="Remote storage URI")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--download", action="store_true", help="Download all models")
    group.add_argument("--upload-models", action="store_true", help="Upload all models")
    group.add_argument("--sync", action="store_true", help="Sync models")
    group.add_argument("--list-local", action="store_true", help="List local models")
    group.add_argument("--list-remote", action="store_true", help="List remote models")

    parser.add_argument("--agent", help="Specific agent to download/upload")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")
    parser.add_argument("--download-data", action="store_true", help="Download data catalog")

    args = parser.parse_args()

    manager = ModelManager(
        local_dir=args.local_dir,
        remote_uri=args.remote_uri,
    )

    if args.list_local:
        models = manager.list_local_models()
        print(f"\nLocal models ({len(models)}):")
        for m in models:
            print(f"  - {m}")

    elif args.list_remote:
        models = manager.list_remote_models()
        print(f"\nRemote models ({len(models)}):")
        for m in models:
            print(f"  - {m}")

    elif args.download:
        if args.agent:
            manager.download_model(args.agent, args.overwrite)
        else:
            downloaded = manager.download_all(args.overwrite)
            print(f"\nDownloaded {len(downloaded)} models")

    elif args.upload_models:
        if args.agent:
            manager.upload_model(args.agent)
        else:
            uploaded = manager.upload_all()
            print(f"\nUploaded {len(uploaded)} models")

    elif args.sync:
        results = manager.sync()
        print(f"\nSync complete:")
        print(f"  Downloaded: {len(results['downloaded'])}")
        print(f"  Uploaded: {len(results['uploaded'])}")

    elif args.download_data:
        print("Data catalog download not implemented")
        print("Use: rsync -avz user@vps:/opt/nautilus-agents/data/catalog ./data/")


if __name__ == "__main__":
    main()
