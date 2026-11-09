import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from Bio import SeqIO
from Bio.Data import CodonTable

#### This script is a comprehensive bioinformatics pipeline designed to analyze 
#### codon usage and codon usage bias by Relative Synonymous Codon Usage (RSCU) calculations
#### in mitochondrial genomes (specifically using the Vertebrate Mitochondrial Genetic Code / NCBI Table 2).  
#### It processes nucleotide sequence data linked to taxonomic metadata, performs statistical tests (Kruskal-Wallis),
#### and generates multi-format visualization plots.


# Suppress minor NumPy/SciPy C-extension system warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# 1. VERTEBRATE MITOCHONDRIAL GENETIC CODE SETUP (NCBI Table 2)
# -----------------------------------------------------------------------------
vert_mito_table = CodonTable.unambiguous_dna_by_id[2]

CODON_TO_AA = dict(vert_mito_table.forward_table)
for stop_codon in vert_mito_table.stop_codons:
    CODON_TO_AA[stop_codon] = "STOP"

AA_TO_CODONS = {}
for codon, aa in CODON_TO_AA.items():
    AA_TO_CODONS.setdefault(aa, []).append(codon)

ALL_CODONS = sorted(list(CODON_TO_AA.keys()))


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS FOR CALCULATING CODON USAGE AND RSCU
# -----------------------------------------------------------------------------
def calculate_codon_counts(sequence):
    """Counts non-overlapping codons in a coding sequence."""
    counts = {codon: 0 for codon in ALL_CODONS}
    seq = str(sequence).upper()
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in counts:
            counts[codon] += 1
    return counts


def calculate_rscu(counts):
    """Calculates Relative Synonymous Codon Usage (RSCU)."""
    rscu = {}
    for aa, codons in AA_TO_CODONS.items():
        total_count = sum(counts[c] for c in codons)
        num_codons = len(codons)
        for c in codons:
            if total_count > 0:
                rscu[c] = (counts[c] * num_codons) / total_count
            else:
                rscu[c] = 0.0
    return rscu


# -----------------------------------------------------------------------------
# 3. PIPELINE FUNCTION
# -----------------------------------------------------------------------------
def run_pipeline(gene_mapping_csv, taxonomy_csv, output_prefix="output"):
    # Load input files with automatic separator detection
    gene_map = pd.read_csv(gene_mapping_csv, sep=None, engine="python")
    tax_map = pd.read_csv(taxonomy_csv, sep=None, engine="python")

    # Clean up column headers: remove leading/trailing spaces and BOM characters
    gene_map.columns = gene_map.columns.str.strip().str.replace('\ufeff', '', regex=False)
    tax_map.columns = tax_map.columns.str.strip().str.replace('\ufeff', '', regex=False)

    # Verify expected column headers exist
    if "Gene" not in gene_map.columns or "Fasta_file_name" not in gene_map.columns:
        raise ValueError(
            f"Columns 'Gene' and 'Fasta_file_name' required in {gene_mapping_csv}. Found: {list(gene_map.columns)}"
        )

    if "Species" not in tax_map.columns or "Genera" not in tax_map.columns or "Subfamily" not in tax_map.columns:
        raise ValueError(
            f"Columns 'Species', 'Genera', and 'Subfamily' required in {taxonomy_csv}. Found: {list(tax_map.columns)}"
        )

    usage_records = []

    # Parse FASTA files mapped by gene_map.csv
    for idx, row in gene_map.iterrows():
        gene_name = str(row["Gene"]).strip()
        fasta_file = str(row["Fasta_file_name"]).strip()

        if not os.path.exists(fasta_file):
            print(f"Warning: File '{fasta_file}' not found. Skipping.")
            continue

        for record in SeqIO.parse(fasta_file, "fasta"):
            header_id = record.id
            counts = calculate_codon_counts(record.seq)
            rscu = calculate_rscu(counts)

            entry = {"Species_ID": header_id, "Gene": gene_name}
            for codon in ALL_CODONS:
                entry[f"Count_{codon}"] = counts[codon]
                entry[f"RSCU_{codon}"] = rscu[codon]

            usage_records.append(entry)

    if not usage_records:
        raise RuntimeError("No sequences processed. Please check your FASTA file paths and gene mapping file.")

    df_usage = pd.DataFrame(usage_records)
    df_usage.to_csv(f"{output_prefix}_codon_usage.csv", index=False)
    print(f"Codon usage table written to {output_prefix}_codon_usage.csv")

    # Match taxonomy metadata
    def find_taxonomy(species_id, df_tax):
        for _, tax_row in df_tax.iterrows():
            sp_name = str(tax_row["Species"]).strip()
            if sp_name and sp_name in species_id:
                return pd.Series([tax_row["Subfamily"], tax_row["Genera"]])
        return pd.Series([np.nan, np.nan])

    df_merged = df_usage.copy()
    df_merged[["Subfamily", "Genera"]] = df_merged["Species_ID"].apply(
        lambda x: find_taxonomy(x, tax_map)
    )

    rscu_cols = [c for c in df_merged.columns if c.startswith("RSCU_")]
    count_cols = [c for c in df_merged.columns if c.startswith("Count_")]

    # -------------------------------------------------------------------------
    # 4. STATISTICAL ANALYSIS (Kruskal-Wallis across Genera AND Subfamilies)
    # -------------------------------------------------------------------------
    stat_results = []
    
    for category in ["Genera", "Subfamily"]:
        for c in rscu_cols:
            codon_name = c.replace("RSCU_", "")
            groups = [
                group[c].dropna().values
                for name, group in df_merged.groupby(category)
                if len(group) >= 2
            ]
            
            if len(groups) > 1:
                try:
                    stat, pval = kruskal(*groups)
                    stat_results.append({
                        "Codon": codon_name,
                        "Taxonomic_Level": category,
                        "Kruskal_H_Stat": stat,
                        "p_value": pval
                    })
                except ValueError:
                    # Occurs if all values in groups are identical
                    continue

    df_stats = pd.DataFrame(stat_results)
    
    # Apply Benjamini-Hochberg FDR correction per taxonomic level
    if not df_stats.empty:
        for level in ["Genera", "Subfamily"]:
            mask = df_stats["Taxonomic_Level"] == level
            if mask.any():
                p_vals = df_stats.loc[mask, "p_value"].values
                _, q_vals, _, _ = multipletests(p_vals, method="fdr_bh")
                df_stats.loc[mask, "adj_p_value_FDR"] = q_vals

    df_stats.to_csv(f"{output_prefix}_statistical_tests.csv", index=False)
    print(f"Statistical test results written to {output_prefix}_statistical_tests.csv")

    # -------------------------------------------------------------------------
    # 5. VISUALIZATIONS
    # -------------------------------------------------------------------------
    # A. General Codon Usage Summary Across All Species
    plt.figure(figsize=(16, 6))
    
    df_species_counts = df_merged.groupby("Species_ID")[count_cols].sum()
    df_species_props = df_species_counts.div(df_species_counts.sum(axis=1), axis=0)
    df_species_props.columns = [c.replace("Count_", "") for c in df_species_props.columns]

    mean_usage = df_species_props.mean()
    std_usage = df_species_props.std()

    sorted_codons = sorted(ALL_CODONS, key=lambda c: (CODON_TO_AA[c], c))
    mean_usage = mean_usage[sorted_codons]
    std_usage = std_usage[sorted_codons]

    x = np.arange(len(sorted_codons))
    plt.bar(x, mean_usage.values, yerr=std_usage.values, capsize=3, color="steelblue", edgecolor="black", alpha=0.85)
    
    plt.xticks(x, [f"{c}\n({CODON_TO_AA[c]})" for c in sorted_codons], rotation=90, fontsize=7)
    plt.ylabel("Mean Proportion of Total Codons")
    plt.title("General Mitochondrial Codon Usage Across All Species (Vertebrate Code)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_general_codon_usage.pdf", format="pdf")
    plt.savefig(f"{output_prefix}_general_codon_usage.svg", format="svg")
    plt.close()

    # B. PCA Plot (Colored by Subfamily & Shaped by Genera)
    from sklearn.decomposition import PCA

    df_rscu_species = df_merged.groupby(["Species_ID", "Genera", "Subfamily"])[rscu_cols].mean().reset_index()

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(df_rscu_species[rscu_cols])
    df_rscu_species["PC1"] = pca_coords[:, 0]
    df_rscu_species["PC2"] = pca_coords[:, 1]

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df_rscu_species,
        x="PC1",
        y="PC2",
        hue="Subfamily",
        style="Genera",
        s=120,
        palette="Dark2",
    )
    plt.title("Codon Bias Analysis: PCA of Mean RSCU Values")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance Explained)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance Explained)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_codon_bias_pca.pdf", format="pdf")
    plt.savefig(f"{output_prefix}_codon_bias_pca.svg", format="svg")
    plt.close()

    # C. Statistical Significance Volcano/Manhattan Plot (Genera Level)
    if not df_stats.empty:
        df_gen = df_stats[df_stats["Taxonomic_Level"] == "Genera"].copy()
        if not df_gen.empty:
            plt.figure(figsize=(10, 6))
            df_gen["neg_log10_fdr"] = -np.log10(df_gen["adj_p_value_FDR"])

            sns.scatterplot(
                data=df_gen,
                x="Kruskal_H_Stat",
                y="neg_log10_fdr",
                hue="adj_p_value_FDR",
                palette="viridis_r",
                size="Kruskal_H_Stat",
                sizes=(40, 220),
            )

            plt.axhline(
                y=-np.log10(0.05),
                color="crimson",
                linestyle="--",
                linewidth=1.2,
                label="FDR Threshold (q = 0.05)",
            )

            top_codons = df_gen.sort_values("Kruskal_H_Stat", ascending=False).head(5)
            for _, r in top_codons.iterrows():
                plt.text(
                    r["Kruskal_H_Stat"] + 1,
                    r["neg_log10_fdr"],
                    r["Codon"],
                    fontsize=9,
                    fontweight="bold",
                )

            plt.title("Mitochondrial Codon Significance Distribution Across Genera", fontsize=12, fontweight="bold")
            plt.xlabel("Kruskal-Wallis H Statistic", fontsize=10)
            plt.ylabel("-log10 (FDR Adjusted p-value)", fontsize=10)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_statistical_significance.pdf", format="pdf")
            plt.savefig(f"{output_prefix}_statistical_significance.svg", format="svg")
            plt.close()

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline(
        gene_mapping_csv="/home/kaleb/Leptodactylidae_mitogenomes/Codon_Bias_Analysis/gene_map.csv",
        taxonomy_csv="/home/kaleb/Leptodactylidae_mitogenomes/Codon_Bias_Analysis/taxonomy_map.csv",
        output_prefix="leptodactylidae_mitochondrial_codon_bias",
    )
