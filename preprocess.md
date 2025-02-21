# Preprocessing Steps
Overview of preprocessing done on iwasaki dataset, with the goal of understanding how 
to generalize this for my iXnos implementation.  

## 1. Shell script (`iwasaki.sh`): 
    - does 2 things: 
        1. Genome-side preprocessing: Using general human genome data as downloaded 
        from iXnos repo, creates bowtie and RSEM indices if they don't exist already. 
        These are necessary for aligning footprint reads to the genome/transcriptome.
        I think this should be the same across datasets associated w a specific 
        species, i.e. if you've already run once for human this won't run again 
        because the necessary index files are already there.  
        2. Dataset-specific: Takes the raw fastq files from your ribo-seq experiment 
        and preprocesses them. 
            - Trimming linkers: This is gonna be specific for each experiment, as the 
            linkers used might be different. If you know the linkers, you'd have to 
            edit `trim_linker.pl` to trim the linkers you used in the ribo-seq 
            experiment.
            - Filter out rRNA and tRNA reads: Filter out rRNA and tRNA reads by using
            bowtie to align reads to rRNA and tRNA reference files. Aligned tRNA and 
            rRNA reads are saved, but I'm not sure if they're ever referred to later...
            - 
```
GENOME_DIR=$1
IWASAKI_PROC_DIR=$2

echo $GENOME_DIR
# Genome-side preprocessing: 
# 1. Prepare a bowtie index if it doesn't already exist
if [ ! -f $GENOME_DIR/human.transcripts.13cds10.1.ebwt ]; then
	bowtie-build $GENOME_DIR/gencode.v22.transcript.13cds10.fa $GENOME_DIR/human.transcripts.13cds10
fi
# 2. Prepare an RSEM index if it doesn't already exist
if [ ! -f $GENOME_DIR/human.transcripts.13cds10.idx.fa ]; then
	echo "I'm making a human index!!!"
	rsem-prepare-reference $GENOME_DIR/gencode.v22.transcript.13cds10.fa $GENOME_DIR/human.transcripts.13cds10
fi
# Dataset-specific preprocessing
# Trim the linkers from the ribo-seq reads datasets and output for all into 1 file
cat $IWASAKI_PROC_DIR/SRR2075925.fastq $IWASAKI_PROC_DIR/SRR2075926.fastq | $IWASAKI_PROC_DIR/trim_linker.pl > $IWASAKI_PROC_DIR/SRR2075925_SRR2075926.trimmed.fastq

# Filter out rRNA and tRNA reads by aligning reads to rRNA or tRNA reference. 
# Unaligned reads (aka reads not from rRNA/tRNA) area saved to 
# iwasaki.not_rrna_trna.fastq. 
# Aligned rRNA and tRNA reads are saved to iwasaki.rrna_trna.sam
bowtie -v 2 -p 36 -S --un $IWASAKI_PROC_DIR/iwasaki.not_rrna_trna.fastq \
	$GENOME_DIR/human_rrna_trna \
	$IWASAKI_PROC_DIR/SRR2075925_SRR2075926.trimmed.fastq > $IWASAKI_PROC_DIR/iwasaki.rrna_trna.sam 2> $IWASAKI_PROC_DIR/iwasaki.rrna_trna.bowtiestats

# Determine footprints by aligning filtered reads to human transcriptome.
# Generates a .sam file. 
bowtie -a --norc -v 2 -p 36 -S --un $IWASAKI_PROC_DIR/iwasaki.unmapped.fastq \
	$GENOME_DIR/human.transcripts.13cds10 \
	$IWASAKI_PROC_DIR/iwasaki.not_rrna_trna.fastq > $IWASAKI_PROC_DIR/iwasaki.footprints.sam 2> $IWASAKI_PROC_DIR/iwasaki.footprints.bowtiestats

# Calculates expression of each gene based on the footprints. 
# This is prob necessary to scale counts
rsem-calculate-expression --sam $IWASAKI_PROC_DIR/iwasaki.footprints.sam $GENOME_DIR/human.transcripts.13cds10 $IWASAKI_PROC_DIR/iwasaki 2> $IWASAKI_PROC_DIR/iwasaki.rsem.stderr
# Converts bam file created above into sam file
samtools view -h $IWASAKI_PROC_DIR/iwasaki.transcript.bam > $IWASAKI_PROC_DIR/iwasaki.transcript.sam
```

## 2. Python Scripts
There are a few additional preprocessing steps that are handled in python scripts.
I should make sure I have these all in python3 format, but they may already be done in
vestigium. 

1. Additional edits on top of the sam file
```
python reproduce_scripts/process_data.py edit_sam_file iwasaki \
	expts/iwasaki expts/iwasaki/process/iwasaki.transcript.sam
```

`mkdir expts/iwasaki/plots` -- might as well handle this in one of the python scripts

2. Determine A-site offsets
    - i.e. given footprints of certain lengths, determine which nt positions correspond
    to the A-site codon. 
```
python reproduce_scripts/process_data.py size_and_frame_analysis iwasaki \
	expts/iwasaki expts/iwasaki/process/iwasaki.transcript.mapped.wts.sam \
	genome_data/gencode.v22.transcript.13cds10.lengths.txt
```
3. More sam file editing...
```
python reproduce_scripts/process_data.py process_sam_file iwasaki \
	expts/iwasaki expts/iwasaki/process/iwasaki.transcript.mapped.wts.sam \
	genome_data/gencode.v22.transcript.13cds10.lengths.txt genome_data/gencode.v22.transcript.13cds10.fa
```