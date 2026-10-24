# Peptide teacher checkpoint

The checkpoint is distributed as a GitHub Release asset because it exceeds
the web uploader's 25 MB file limit. Open the
[APRnet v1.0.0 Release](https://github.com/HPF66/APRNet/releases/tag/v1.0.0)
or use the [direct checkpoint download](https://github.com/HPF66/APRNet/releases/download/v1.0.0/teacher_canya_cpad.pt)
and place the file at the following path relative to the repository root:

```text
teacher/pretrained/teacher_canya_cpad.pt
```

Cloning the repository or downloading its source ZIP does not include Release
assets. From the repository root on Linux, download and check the file with:

```bash
curl --fail --location --output teacher/pretrained/teacher_canya_cpad.pt \
  https://github.com/HPF66/APRNet/releases/download/v1.0.0/teacher_canya_cpad.pt
sha256sum teacher/pretrained/teacher_canya_cpad.pt
```

On Windows, verify a downloaded file with:

```powershell
(Get-FileHash "teacher/pretrained/teacher_canya_cpad.pt" -Algorithm SHA256).Hash.ToLower()
```

Expected SHA-256:

```text
7b059283e57e54dcff3e48ad1e339deb7251aedc55b977d0bd79a489aa35b327
```

The file is the unchanged CAnYA shared-supervised -> CPAD final teacher. It is
a trained model checkpoint, not generated HAPR or ESM-2 feature data. Teacher
inference still needs compatible ESM-2 8M backbone assets. CAnYA pretraining
additionally needs its original initializer.

The file size is 36,065,562 bytes. See `metadata.json` for the recorded file
identity and [the encoder compatibility note](../../encoder/pretrained/README.md)
before using this teacher to prepare encoder inputs.
