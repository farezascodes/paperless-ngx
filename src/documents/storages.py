"""
Object storage backend for document media.

Only the ``documents`` alias in ``settings.STORAGES`` resolves to this module.
``default`` and ``staticfiles`` are deliberately untouched, so enabling object
storage never changes how the app logo or the whitenoise-compressed static
files are served.

This module stays deliberately thin. Anything expressible as django-storages
configuration belongs in ``STORAGES["documents"]["OPTIONS"]`` (see
``paperless.settings``), not here: retries, timeouts, addressing style,
signature version and checksum behaviour are all declarative.

The subclass exists only for the three capabilities django-storages does not
provide. Each was verified against the installed package rather than assumed:

1. **Server-side copy.** ``copy_object``/``CopySource`` appear nowhere in
   ``storages/``, so a rename would otherwise mean downloading the whole
   object and re-uploading it.
2. **A local filesystem path.** ``Storage.path()`` raises
   ``NotImplementedError`` (``django/core/files/storage/base.py:131``) and
   ``S3Storage`` does not override it, yet every parser, every ``pikepdf``
   call and the pre/post-consume scripts need a real file on disk.
3. **Batch delete.** django-storages issues one request per key.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from django.conf import settings
from storages.backends.s3 import S3Storage
from storages.utils import clean_name

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

# S3 DeleteObjects accepts at most 1000 keys per request.
DELETE_BATCH_SIZE = 1000


class S3DocumentStorage(S3Storage):
    """S3-compatible storage for originals, archive files and thumbnails."""

    def key_for(self, name: str) -> str:
        """Return the fully-qualified object key a storage name maps to."""
        return self._normalize_name(clean_name(name))

    def copy(self, old_name: str, new_name: str) -> None:
        """
        Server-side copy. Never transfers bytes through this process.

        boto3's managed ``copy`` transparently switches to multipart for large
        objects, so this is safe regardless of document size.
        """
        self.bucket.copy(
            {"Bucket": self.bucket_name, "Key": self.key_for(old_name)},
            self.key_for(new_name),
        )

    def move(self, old_name: str, new_name: str) -> None:
        """
        Rename an object.

        S3 has no atomic rename, so this is copy-then-delete and the source is
        removed only after the copy returns. A failure between the two leaves
        the source intact, which is the recoverable direction.
        """
        self.copy(old_name, new_name)
        self.delete(old_name)

    def delete_many(self, names: Iterable[str]) -> None:
        """Delete many keys, batching to stay within the DeleteObjects limit."""
        keys = [{"Key": self.key_for(name)} for name in names]
        for start in range(0, len(keys), DELETE_BATCH_SIZE):
            self.bucket.delete_objects(
                Delete={"Objects": keys[start : start + DELETE_BATCH_SIZE]},
            )

    @contextmanager
    def local_path(self, name: str, *, suffix: str = "") -> Iterator[Path]:
        """
        Materialise an object as a real file and yield its path.

        Parsers, ``pikepdf`` and the consume scripts all require a filesystem
        path, so this is the bridge between object storage and the rest of
        paperless. The download is a boto3 managed transfer, so large objects
        stream to disk rather than through memory.

        The file is created inside ``SCRATCH_DIR`` and removed on exit,
        including when the caller raises. Pass ``suffix`` when the consumer
        infers a type from the extension, which most parsers do.
        """
        scratch: Path = settings.SCRATCH_DIR
        scratch.mkdir(parents=True, exist_ok=True)

        handle, raw_path = tempfile.mkstemp(dir=scratch, suffix=suffix)
        os.close(handle)
        path = Path(raw_path)
        try:
            self.bucket.Object(self.key_for(name)).download_file(str(path))
            yield path
        finally:
            path.unlink(missing_ok=True)
