import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

#### Pipeline for analyzing start and stop codon usage among mitochondrial gene alignments, 
#### configured to use NCBI Table 2 (vertebrate mitochondria). It scans a directory of sequence alignments,
#### parses the genes, extract their start and stop codons, audits the data, computes frequence/percentage tables,
#### and generates a stacked bar chart figure

ALIGNMENTS_DIR = '~/path_to_your_files' #### changes this command to direct to your directory
OUTPUT_DIR = '.' #### this command may also be changed
# False is conservative for potentially truncated alignment sequences.
INCLUDE_INCOMPLETE_STOPS = False

GENE_MAP = {
    'atp6': 'atp6', 'at6': 'atp6', 'atp8': 'atp8', 'at8': 'atp8',
    'cox1': 'cox1', 'co1': 'cox1', 'cx1': 'cox1',
    'cox2': 'cox2', 'co2': 'cox2', 'cx2': 'cox2',
    'cox3': 'cox3', 'co3': 'cox3', 'cx3': 'cox3',
    'cytb': 'cytb', 'cob': 'cytb', 'cb': 'cytb',
    'nad1': 'nad1', 'nd1': 'nad1', 'nad2': 'nad2', 'nd2': 'nad2',
    'nad3': 'nad3', 'nd3': 'nad3', 'nad4': 'nad4', 'nd4': 'nad4',
    'nad4l': 'nad4L', 'nd4l': 'nad4L', 'nad5': 'nad5', 'nd5': 'nad5',
    'nad6': 'nad6', 'nd6': 'nad6'
}
GENES_ORDER = ['nad1', 'nad2', 'nad3', 'nad4', 'nad4L', 'nad5', 'nad6',
               'atp6', 'atp8', 'cox1', 'cox2', 'cox3', 'cytb']
START_CODONS = ['ATT', 'ATC', 'ATA', 'ATG', 'GTG']  # NCBI table 2
STOP_CODONS = ['TAA', 'TAG', 'AGA', 'AGG']         # NCBI table 2


def parse_fasta(path):
    header, chunks = None, []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, ''.join(chunks)


def extract_gene_name(filename):
    name = os.path.basename(filename).lower()
    for alias in sorted(GENE_MAP, key=len, reverse=True):
        if re.search(r'(?:^|_)' + re.escape(alias) + r'(?:_|\.|\b)', name):
            return GENE_MAP[alias]
    return None


def extract_species_name(header):
    return re.sub(r'^\d+_', '', re.sub(r'^\d+_\d+_', '', header.strip()))


def get_stop_codon(seq):
    # Complete codons must occupy the last three nucleotides.
    if len(seq) >= 3 and seq[-3:] in STOP_CODONS:
        return seq[-3:]
    if not INCLUDE_INCOMPLETE_STOPS:
        return None
    # A partial stop must occupy the final reading-frame position, not
    # merely be the final 1-2 bases of an otherwise full-length codon.
    remainder = len(seq) % 3
    if remainder == 1 and seq.endswith('T'):
        return 'T--'
    if remainder == 2 and seq.endswith('TA'):
        return 'TA-'
    return None


def process_alignments(folder_path):
    starts, stops, audit = [], [], []
    paths = sorted({p for extension in ('*.fas', '*.fasta', '*.fa')
                    for p in glob.glob(os.path.join(folder_path, extension))})
    for path in paths:
        gene = extract_gene_name(path)
        if gene is None:
            print(f'Skipping unrecognized filename: {path}')
            continue
        for header, raw in parse_fasta(path):
            species = extract_species_name(header)
            seq = re.sub(r'[-.\s]', '', raw).upper().replace('U', 'T')
            first = seq[:3] if len(seq) >= 3 else ''
            start = first if first in START_CODONS else None
            stop = get_stop_codon(seq)
            if start:
                starts.append({'Gene': gene, 'Species': species, 'Codon': start})
            if stop:
                stops.append({'Gene': gene, 'Species': species, 'Codon': stop})
            audit.append({'File': os.path.basename(path), 'Gene': gene,
                          'Species': species, 'Length_ungapped': len(seq),
                          'First_triplet': first, 'Last_triplet': seq[-3:] if len(seq) >= 3 else '',
                          'Start_counted': bool(start), 'Start_codon': start or '',
                          'Stop_counted': bool(stop), 'Stop_codon': stop or ''})
    return (pd.DataFrame(starts, columns=['Gene', 'Species', 'Codon']),
            pd.DataFrame(stops, columns=['Gene', 'Species', 'Codon']),
            pd.DataFrame(audit))


def make_frequency_table(data, codons):
    counts = pd.crosstab(data['Gene'], data['Codon']) if not data.empty else pd.DataFrame()
    counts = counts.reindex(index=GENES_ORDER, columns=codons, fill_value=0)
    totals = counts.sum(axis=1)
    percentages = counts.div(totals.where(totals.ne(0)), axis=0).fillna(0) * 100
    return counts, percentages, totals


def plot_figure(start_data, stop_data, output_file='Figure1_codon_usage.png'):
    stop_order = STOP_CODONS + (['T--', 'TA-'] if INCLUDE_INCOMPLETE_STOPS else [])
    start_counts, start_pct, start_n = make_frequency_table(start_data, START_CODONS)
    stop_counts, stop_pct, stop_n = make_frequency_table(stop_data, stop_order)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, counts, pct in [('start', start_counts, start_pct), ('stop', stop_counts, stop_pct)]:
        counts.to_csv(os.path.join(OUTPUT_DIR, f'{name}_codons_counts.csv'))
        pct.to_csv(os.path.join(OUTPUT_DIR, f'{name}_codons_usage_table.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ['#1f77b4', '#ff7f0e', '#7f7f7f', '#bcbd22', '#63393E', '#17becf']
    for ax, table, totals, title, panel in [
        (axes[0], start_pct, start_n, 'Start codons usage', 'A'),
        (axes[1], stop_pct, stop_n, 'Stop codons usage', 'B')
    ]:
        table.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(table.columns)],
                   width=0.65, edgecolor='none')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter('{x:.0f}%')
        ax.set_xticklabels(GENES_ORDER, rotation=45, ha='right', fontstyle='italic', fontsize=10)
        ax.set_xlabel('')
        ax.text(-0.08, 1.05, panel, transform=ax.transAxes, fontsize=20, fontweight='bold')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.21), ncol=4,
                  frameon=False, fontsize=9)
        for i, n in enumerate(totals):
            ax.text(i, 101, f'{n}', ha='center', va='bottom', fontsize=7, clip_on=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUTPUT_DIR, output_file)
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(os.path.splitext(png)[0] + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Plots saved: {png} and PDF. Numbers above bars are valid sequence counts.')


if __name__ == '__main__':
    df_start, df_stop, audit = process_alignments(ALIGNMENTS_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audit.to_csv(os.path.join(OUTPUT_DIR, 'codon_validation_audit.csv'), index=False)
    print(f'Valid starts: {len(df_start)}; valid stops: {len(df_stop)}; sequences checked: {len(audit)}')
    if audit.empty:
        print('No sequences found. Check ALIGNMENTS_DIR and gene filenames.')
    else:
        plot_figure(df_start, df_stop)
