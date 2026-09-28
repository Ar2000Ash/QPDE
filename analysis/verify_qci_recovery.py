#!/usr/bin/env python3
"""Reassemble and independently verify the archived 2026-09-26 QCI recovery package.

Reconstruction is offline: no vendor API or credentials are required. The historical
polynomial-upload bytes are NOT in this package; the 34 mathematical QUBOs are
reconstructions and have different file SHA256 values from historic term hashes.
"""
from __future__ import annotations
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / 'data' / 'provenance' / 'qci_reconstruction_2026_09_26' / 'parts'
ZIP_HASH = '2cfd274f76e0302570042a6ed071cfc0bcc18906f03c30e56a85554cddf4a3a1'
PART_SHA1 = (
 'cb8f8112adb02ddc4120e2ed9d700101f3a430b9',
 '776288bb8ac4718bd8870789e075c7ed7f0ae7a4',
 '5b79b5cc251fcce7edbdfc4b94b70a49318c0b5c',
 '7e346e4ba977e787dd9822a88f3bb4762e8dc647',
 '9146f36b17b1fd609075b9acab5f5beedc62632f',
 'cb4aa3a3d3deb763ebca3ef378567b48a9ebba0d',
 'c9b3d6214e80ab1432cd711e0d5bde19804e4743',
 '6eacedf639aaf9a618598aa94fbc7b4fd3ca727e',
 'ec7c307652aaba3302decdceb9eddcb0a930746a',
)


def validate_manifest(base: Path, sources_only: bool = False) -> None:
    entries = [line.strip() for line in (base/'SHA256SUMS.txt').read_text().splitlines() if line.strip()]
    for line in entries:
        digest, rel = line.split(maxsplit=1)
        rel = rel.removeprefix('./')
        if sources_only and not rel.startswith('source/'):
            continue
        path = base / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError(f'Recovery archive checksum mismatch: {rel}')


def main() -> None:
    files = sorted(PARTS.glob('recovery.zip.part*'))
    if [f.name for f in files] != [f'recovery.zip.part{i:02d}' for i in range(9)]:
        raise AssertionError('Expected the nine ordered immutable recovery ZIP parts')
    with tempfile.TemporaryDirectory(prefix='qpde_qci_recovery_') as temp:
        dest = Path(temp)
        archive = dest/'recovery.zip'
        digest = hashlib.sha256()
        with archive.open('wb') as out:
            for idx, part in enumerate(files):
                block = part.read_bytes()
                object_sha = hashlib.sha1(f'blob {len(block)}\0'.encode()+block).hexdigest()
                if object_sha != PART_SHA1[idx]:
                    raise AssertionError(f'Wrong recovery part: {part.name}')
                out.write(block)
                digest.update(block)
        if digest.hexdigest() != ZIP_HASH or archive.stat().st_size != 394421:
            raise AssertionError('Original recovery archive SHA256/size mismatch')
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise AssertionError('Corrupted recovery ZIP member')
            for member in z.infolist():
                target = (dest / member.filename).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise AssertionError(f'Unsafe ZIP member: {member.filename}')
            z.extractall(dest)
        pkg = dest/'qpde_qci_end_to_end_recovery'
        validate_manifest(pkg)
        for name in ('verify_package.py', 'reconstruct_and_audit.py', 'verify_package.py'):
            subprocess.run([sys.executable,str(pkg/'scripts'/name)],cwd=pkg,check=True)
        # Derived floating-point files can vary in their last digits across NumPy
        # builds: verify immutable source hashes, then compare numerical outputs via
        # verify_package.py (already rerun above), rather than demanding identical
        # derived JSON/CSV bytes across operating systems.
        validate_manifest(pkg, sources_only=True)
        report = (pkg/'AUDIT_REPORT.json').read_text()
        import json
        data = json.loads(report)
        assert (data['mathematical_QUBOs_reconstructed'],data['mapped_instances'],
                data['unique_hardware_returned_sample_states'],
                data['original_sample_rows_with_wrong_signed_decoding_or_residual']) == (34,60,457,136)
        assert data['reconstructed_byte_hashes_matching_historical_hashes']==0
        print('PASS: original recovery ZIP, full package hashes, 34 mathematical QUBOs, 457 raw samples and offline regeneration verified.')


if __name__ == '__main__':
    main()