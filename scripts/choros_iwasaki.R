library(here)
library(choros)

set.seed(10)

# Location of new-to-choros files
choros_dir <- file.path(here(), "processed-data/choros_iwasaki")
offsets_fname <- file.path(choros_dir, "Asite_rules.txt")
# gtf_fname <- file.path(choros_dir, "gencode.v22.annotation.gtf")

# reference files
ref_dir <- file.path(here(), "iXnos/genome_data")
transcript_fa_fname <- file.path(ref_dir, "gencode.v22.transcript.13cds10.fa")
transcript_length_fname <- file.path(ref_dir, "gencode.v22.transcript.13cds10.lengths.txt")
transcript_lengths <- load_lengths(transcript_length_fname)

# experiment-specific files
expt <- "iwasaki"
dataset_dir <- file.path(here(), "iXnos/expts/iwasaki/process")
bam_alignment_fname <- file.path(dataset_dir, "iwasaki.transcript.bam")

# Define output dir
results_dir <- file.path(choros_dir, "outputs")
if(!dir.exists(results_dir)) { dir.create(results_dir) }


# choros parameters
min_prop <- 0.9
f5_length <- 3
f3_length <- 3
# num_genes <- 250
num_genes <- 150
min_coverage <- 5
min_nonzero <- 100
regression_model <- formula(count ~ transcript + A + P + E + d5*f5 + d3*f3 + gc)

# generate diagnostic plot ------------------------------------------------
print("Generating diagnostic plot...")
diagnostic_plot_obj <- paste0(expt, "_diagnostic_plot")
diagnostic_plot_fname <- file.path(results_dir, paste0(diagnostic_plot_obj, ".Rda"))
if(!file.exists(diagnostic_plot_fname)) {
  assign(diagnostic_plot_obj,
         plot_diagnostic(bam_alignment_fname, transcript_length_fname))
  save(list=diagnostic_plot_obj, file=diagnostic_plot_fname)
}

# load footprint alignments -----------------------------------------------
print("Loading footprint alignments...")
bam_obj <- paste0(expt, "_bam")
bam_fname <- file.path(results_dir, paste0(bam_obj, ".Rda"))
if(!file.exists(bam_fname)) {
  assign(bam_obj,
         load_bam(bam_alignment_fname, transcript_fa_fname,
                  transcript_length_fname, offsets_fname,
                  f5_length=f5_length, f3_length=f3_length))
  save(list=bam_obj, file=bam_fname)
} else {
  load(bam_fname)
}

# compute size/frame subsets ----------------------------------------------
print("Computing size/frame subsets")
d5_d3_obj <- paste0(expt, "_d5_d3")
d5_d3_fname <- file.path(results_dir, paste0(expt, "_d5_d3.Rda"))
subsets_obj <- paste0(expt, "_subsets")
subsets_fname <- file.path(results_dir, paste0(expt, "_subsets.Rda"))
if(!file.exists(subsets_fname)) {
  assign(d5_d3_obj, count_d5_d3(get(bam_obj)))
  save(list=d5_d3_obj, file=d5_d3_fname)
  assign(subsets_obj, choose_subsets(get(d5_d3_obj)))
  save(list=subsets_obj, file=subsets_fname)
} else {
  load(subsets_fname)
}

# choose transcripts for training/test sets -------------------------------
print("Choosing transcripts for training/test sets")
# per transcript: calculate mean codon coverage and number nonzero codon positions
transcript_coverage <- calculate_transcript_density(get(bam_obj),
                                                    transcript_length_fname,
                                                    exclude_codons5=0, exclude_codons3=0)
transcript_num_nonzero <- count_nonzero_codons(get(bam_obj))

# check if number good transcripts sufficient for training
good_transcripts <- data.frame(transcript=names(transcript_coverage),
                               coverage=transcript_coverage,
                               num_nonzero=transcript_num_nonzero[match(names(transcript_coverage),
                                                                        names(transcript_num_nonzero))])
good_transcripts <- subset(good_transcripts,
                           coverage > min_coverage & num_nonzero > min_nonzero)

# PARALOG STUFF -- commenting out bc I didn't use in my iXnos runs
# paralogs_fname <- file.path(ref_dir, "scer_100_80.paralogs.id.txt")
# # identify paralogs
# paralogs <- readLines(paralogs_fname)
# paralogs <- strsplit(paralogs, split="\t")
# paralogs <- subset(paralogs, lengths(paralogs) > 1)
# paralogs <- data.frame(transcript=unlist(paralogs),
#                        group=unlist(mapply(rep, seq(length(paralogs)), times=lengths(paralogs))))
# good_transcripts$paralog_group <- paralogs$group[match(good_transcripts$transcript,
#                                                        paralogs$transcript)]

# # take top transcript by read coverage per paralog group
# paralog_groups <- split(good_transcripts, good_transcripts$paralog_group)
# paralog_groups <- lapply(paralog_groups,
#                          function(x) {
#                            if(nrow(x) == 1) {
#                              return(x)
#                            } else {
#                              x <- x[order(x$coverage, decreasing=T),]
#                              return(x[1, ])
#                            }
#                          })
# paralog_groups <- do.call(rbind, paralog_groups)
# good_transcripts <- rbind(subset(good_transcripts,
#                                  is.na(good_transcripts$paralog_group)),
#                           paralog_groups)

# take top n=2*num_genes for training/test sets
good_transcripts <- good_transcripts[order(good_transcripts$coverage, decreasing=T),]
if(nrow(good_transcripts) > 2*num_genes) {
  good_transcripts <- good_transcripts[seq(2*num_genes),]
} else {
  num_genes <- round(nrow(good_transcripts)/2)
  print(paste("Insufficient genes for training; new number of genes for training:", num_genes))
}

# split transcripts evenly into training/test sets
training_set <- sample.int(nrow(good_transcripts), size=num_genes)
training_set <- as.character(good_transcripts$transcript)[training_set]
test_set <- subset(good_transcripts, !(as.character(transcript) %in% training_set))
test_set <- as.character(test_set$transcript)

writeLines(training_set, con=file.path(results_dir, "training_set.txt"))
writeLines(test_set, con=file.path(results_dir, "test_set.txt"))

# initialize training data for regression ---------------------------------
print("Initializing training data for regression")
training_obj <- paste0(expt, "_training")
training_fname <- file.path(results_dir, paste0(training_obj, ".Rda"))
if(!file.exists(training_fname)) {
  assign(training_obj,
         init_data(transcript_fa_fname, transcript_length_fname,
                   d5_d3_subsets=get(subsets_obj),
                   f5_length=f5_length, f3_length=f3_length,
                   which_transcripts=training_set))
  assign(training_obj,
         within(get(training_obj),
                count <- count_footprints(get(bam_obj), get(training_obj), "count")))
  save(list=training_obj, file=training_fname)
} else {
  load(training_fname)
}

# compute regression ------------------------------------------------------
print("Computing regression")
fit_obj <- paste0(expt, "_fit")
fit_fname <- file.path(results_dir, paste0(fit_obj, ".Rda"))
coef_obj <- paste0(expt, "_coef")
coef_fname <- file.path(results_dir, paste0(coef_obj, ".Rda"))
if(!file.exists(coef_fname)) {
  assign(fit_obj, glm.nb(regression_model, data=get(training_obj)))
  save(list=fit_obj, file=fit_fname)
  print("Object fit!")
  assign(coef_obj, parse_coefs(get(fit_obj)))
  save(list=coef_obj, file=coef_fname)
} else {
  load(coef_fname)
}
print("Regression done!")
# correct counts ----------------------------------------------------------

correction_name <- paste0("correct_", num_genes)

assign(bam_obj,
       within(get(bam_obj),
              assign(correction_name,
                     correct_bias(get(bam_obj), fit_coefs=get(coef_obj)))))
save(list=bam_obj, file=bam_fname)

assign(training_obj,
       within(get(training_obj),
              assign(correction_name,
                     count_footprints(get(bam_obj), get(training_obj),
                                      correction_name))))
save(list=training_obj, file=training_fname)

# evaluate bias -----------------------------------------------------------

to_evaluate <- c("count", correction_name)
test_data <- subset(get(bam_obj), transcript %in% test_set)

# calculate codon position importance
codon_corr_obj <- paste0(expt, "_codon_corr")
codon_corr_fname <- file.path(results_dir, paste0(codon_corr_obj, ".Rda"))
if(!file.exists(codon_corr_fname)) {
  assign(codon_corr_obj,
         lapply(to_evaluate,
                function(x) {
                  evaluate_bias(test_data, which_column=x,
                                transcript_fa_fname, transcript_length_fname,
                                type="codon")
                }))
  save(list=codon_corr_obj, file=codon_corr_fname)
}

# calulate nt position importance
nt_corr_obj <- paste0(expt, "_nt_corr")
nt_corr_fname <- file.path(results_dir, paste0(nt_corr_obj, ".Rda"))
if(!file.exists(nt_corr_fname)) {
  assign(nt_corr_obj,
         lapply(to_evaluate,
                function(x) {
                  evaluate_bias(test_data, which_column=x,
                                transcript_fa_fname, transcript_length_fname,
                                type="nt")
                }))
  save(list=nt_corr_obj, file=nt_corr_fname)
}

q(save="no")