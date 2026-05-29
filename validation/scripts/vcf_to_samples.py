import cyvcf2
import tsinfer
from pathlib import Path

simpref = "l25kb"   # -> ../../vcf/sim_l1mb_0.vcf  (no trailing underscore)
vcf_path = f"../vcf/sim_{simpref}_0.vcf"
samples_path = f"../vcf/sim_{simpref}_0.samples"

def add_diploid_sites(vcf, samples):
    pos = 0
    for variant in vcf:
        if pos == variant.POS:
            raise ValueError("Duplicate position", pos)
        pos = variant.POS
        if any(not phased for _, _, phased in variant.genotypes):
            raise ValueError("Unphased at", pos)
        alleles = [variant.REF] + variant.ALT
        ancestral = variant.INFO.get("AA", variant.REF)
        ordered = [ancestral] + [allele for allele in alleles if allele != ancestral]
        allele_index = {i: ordered.index(allele) for i, allele in enumerate(alleles)}
        genotypes = [
            allele_index[old]
            for row in variant.genotypes
            for old in row[0:2]
        ]
        samples.add_site(pos, genotypes=genotypes, alleles=ordered, ancestral_allele=0)

vcf = cyvcf2.VCF(vcf_path)
seq_len = vcf.seqlens[0]

with tsinfer.SampleData(path=samples_path, sequence_length=seq_len) as samples:
    add_diploid_sites(vcf, samples)

print(samples_path, samples.num_samples, "haplotypes,", samples.num_sites, "sites")
