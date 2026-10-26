# Paper-to-artifact map

Numbering follows the final 12-page manuscript, *Same Product, Different Visibility: Attribute-Order Sensitivity in Dense Retrieval*, including Appendices A-D.

| Paper item | Evidence and implementation | Reproduced output |
|---|---|---|
| Figure 1: teal chair, GTE, rank 11 to 939 | `data/cases/teal_chair/`; WANDS/GTE target ranks | `figure1.csv`; `figure1_target_ranks.png` |
| Figure 2: interventions, states, and membership | `src/states.py`, `src/analysis.py`, exact target insertion | Conceptual definitions |
| Figure 3: five-encoder target-only sensitivity | Ten `data/target_only/` files; complete native token lengths | `figure3.csv`; `figure3_target_sensitivity.png` |
| Figure 4; Appendix B.2: adjacent swaps | `data/adjacent/`; frozen cohort, swaps, and repeats | `figure4.csv`; `figure4_adjacent_swaps.png` |
| Figure 5: exact within-query cancellation | Source/reversal highest-label identities | `figure5.csv`; `figure5_recall_cancellation.png` |
| Figure 6; Appendix C.1: six source-order contrasts | All five models' catalog ranks and relevant identities | `figure6.csv`; `figure6_membership_changes.png` |
| Figure 7; Appendix D.2: canonical Recall differences | Query-paired canonical ranks versus seven-order means | `figure7.csv`; `figure7_canonical_recall.png` |
| Table 1; Appendix A.4: control definitions | `src/representations.py`, `src/retrieval.py` | Definitions; no separate empirical run |
| Table 2: wooden chair outdoor | All 24 products in the Top-20 union and recorded frame evidence | `table2_products.csv`; `table2_metrics.csv` |
| Table 3: canonical dense and broader controls | 154 rank conditions; direct graded gains | `table3.csv`, with panel A/B labels |
| Table 4; Appendix B.1: five-encoder VI | Full and fully-fitting supports; both interventions | `table4.csv` |
| Table 5: expanded order families | Harmonized ranks and frozen common supports | `table5.csv`; `harmonized_increments.csv` |
| Table 6: two additional models under reversal | Recomputed gains/losses and paired Recall/nDCG differences | `table6.csv` |
| Table 7; Appendix D.1: canonical transitions | Raw states and fixed canonical product identities | `table7.csv`; `transitions_complete.csv` |
| Table 8: hybrid versus dense at K=100 | Paired query-level Recall and VI | `table8.csv` |
| Appendix A.2: native model and boundary audits | `configs/recent_models.json`; `data/boundary/` | `qa/recent_model_boundary_checks.json` |
| Appendix C.2: graded evaluation | All judged ranks and labels; standard and condensed nDCG | `data/effectiveness_per_query.parquet` |

Tables are under `outputs/reproduced/tables/`, figures under `outputs/reproduced/figures/`, and consistency checks under `outputs/reproduced/qa/`. CSV estimates retain full precision; figure/table presentation may be simplified.

The anti-fatigue-mat example is reconstructed from all relevant identities in `anti_fatigue_mat.csv`. The wooden-chair audit preserves unknown and ambiguous frame evidence and does not replace benchmark relevance judgments. Table 7 transition percentages use all highest-label support within each query, not only crossing products.
