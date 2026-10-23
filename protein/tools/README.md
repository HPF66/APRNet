# Protein split generation

The fixed cohort partitions are generated artifacts and are intentionally not
distributed with this repository. Keep source workbooks and source tables
outside the code checkout, then run the project split builders with an explicit
input and output directory. The model loader accepts any generated directory
that follows the contract in `protein/README.md`:

```text
<cohort>/<protocol>/cohort_membership.tsv
<cohort>/<protocol>/outer_foldN/{train,val,test}.tsv
```

Do not place fold outputs, embeddings, graph caches or checkpoints under the
repository. Keep one output directory per cohort and protocol so hashes and
training results remain auditable.
