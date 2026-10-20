# Paper-to-artifact map

This map uses the numbering in the supplied 12-page PDF titled *Same Product, Different Visibility: Attribute-Order Sensitivity in Dense Retrieval*.

| Paper item | Retained evidence | Reproduced output |
|---|---|---|
| Figure 1: teal chair, GTE, rank 11 to 939 | `data/cases/teal_chair/`; `data/target_only/wands_gte_modernbert.parquet` | `figure1.csv`; `figure1_target_ranks.png` |
| Figure 2: intervention/state/membership definitions | Equations implemented in `src/states.py`, `src/analysis.py`, and `scripts/run_retrieval.py` | Conceptual diagram; no empirical data |
| Figure 3; RQ1 primary and fully-fitting sensitivity | Six `data/target_only/` files, `data/token_lengths.csv`, eligible query IDs | `inclusion_complete.csv`; `figure3.csv`; `figure3_target_sensitivity.png` |
| Figure 4; Appendix D.7: adjacent swaps | `data/adjacent/`, including frozen cohort, complete variants, and repeated checks | `figure4.csv`; `data/aggregate_estimates.csv`; `figure4_adjacent_swaps.png` |
| Figure 5; RQ2 Recall cancellation | Raw all-judged ranks in `data/ranks/`; source versus reversed order | `membership_all_contrasts.csv`; `figure5.csv`; `figure5_recall_cancellation.png` |
| Table 1; Appendix D.1-D.3: control definitions | `src/representations.py`, `src/retrieval.py`, `scripts/run_retrieval.py` | Definitions; no separate empirical input |
| Table 2: wooden chair outdoor | `data/cases/wooden_chair/`, full records in WANDS catalog, q367 judgments/ranks | `table2_products.csv`; `table2_metrics.csv` |
| Table 3: effectiveness and instability | Retained all-judged method ranks; query support; corrected graded gains | `table3.csv`; `controls_complete.csv` |
| Table 4: WANDS/MiniLM ascending transitions | Raw seven-schedule and canonical product identities | `table4.csv`; `transitions_complete.csv` |
| Table 5: all canonical transitions | Same identity-level construction for both datasets and encoders | `table5.csv` |
| Figure 6: hybrid minus dense at K=100 | Query-paired control records; 10,000 bootstrap samples | `figure6.csv`; `figure6_hybrid_contrasts.png` |
| Table 6; Appendix D.5: expanded orders | `data/expanded/`; fixed common supports and harmonized ranks | `table6.csv`; `harmonized_summary.csv`; `harmonized_increments.csv` |
| Table 7; Appendix D.6: graded evaluation | All retained labels/ranks; standard and condensed gain handling | `table7.csv`; `data/effectiveness_per_query.parquet` |
| Figure 7; Appendix E: French molding | `data/cases/french_molding/`; q359/p12575 target ranks | `figure7_attributions.csv`; `figure7_french_molding.png` |

Output tables are under `outputs/reproduced/tables/`; plot files are under `outputs/reproduced/figures/`. Plot layouts are simplified, while their estimates and uncertainty intervals are reconstructed from retained evidence.

The RQ2 anti-fatigue-mat example is reconstructed from all relevant IDs, not from a selected list of changed products. The wooden-chair case retains all 24 products in the union of both Top-20 lists, including missing and ambiguous frame-material evidence. Original benchmark labels are not replaced by the manual frame-material audit.
