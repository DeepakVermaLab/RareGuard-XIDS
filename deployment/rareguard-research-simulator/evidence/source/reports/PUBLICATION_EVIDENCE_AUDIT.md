# RareGuard-XIDS Publication Evidence Audit

Status: **NOT_READY**

## Readiness Gates

| gate | passed |
| --- | --- |
| all_data_splits_ready | True |
| validation_only_tuning_complete | True |
| selected_hyperparameters_frozen | True |
| deterministic_confirmatory_device_match | True |
| central_complete | True |
| classical_baselines_complete | True |
| continual_complete | True |
| federated_complete | True |
| agentic_complete | True |
| agent_validation_selection_complete | True |
| robustness_complete | True |
| confirmation_rareguard_robustness_superiority | False |
| confirmation_locked_test_once_per_preregistered_run | True |
| confirmation_rareguard_mean_superiority_macro_and_rare | False |
| full_model_beats_all_ablations_primary_composite | False |
| continual_memory_beats_replay_controls | False |
| agent_improves_over_light_with_partial_escalation | False |
| joule_energy_claim_supported | False |
| full_corpus_reproduction | False |

## Paired Seed Gains

| dataset | metric | paired_seeds | mean_gain_rareguard_minus_nova | ci95_lower | ci95_upper |
| --- | --- | --- | --- | --- | --- |
| cic_bccc_nrc_tabulariotattack_2024 | f1_macro | 5 | 0.03473034246310418 | -0.0008823369933686834 | 0.07034302191957703 |
| cic_bccc_nrc_tabulariotattack_2024 | rare_f1_macro | 5 | 0.0017554984349524155 | -0.042715568682464 | 0.04622656555236883 |
| cic_bccc_nrc_tabulariotattack_2024 | balanced_accuracy | 5 | 0.0067204954894817485 | -0.028970413876484066 | 0.04241140485544756 |
| cic_bccc_nrc_tabulariotattack_2024 | mcc | 5 | 0.03326763564600414 | -0.0015359906584197155 | 0.06807126195042801 |
| cic_bccc_nrc_tabulariotattack_2024 | ece_15 | 5 | -0.016935328874393103 | -0.02567146042560747 | -0.008199197323178736 |
| cic_iiot_2025_datasense | f1_macro | 5 | 0.4761042216686536 | 0.4452918713331267 | 0.5069165720041805 |
| cic_iiot_2025_datasense | rare_f1_macro | 5 | 0.4559791689330467 | 0.41927282171562974 | 0.49268551615046363 |
| cic_iiot_2025_datasense | balanced_accuracy | 5 | 0.26155327904766656 | 0.22754676751286085 | 0.2955597905824723 |
| cic_iiot_2025_datasense | mcc | 5 | 0.5513502447986631 | 0.5179406217997176 | 0.5847598677976087 |
| cic_iiot_2025_datasense | ece_15 | 5 | -0.08864904634539897 | -0.11272197024380687 | -0.06457612244699107 |
| ciciot2023 | f1_macro | 5 | 0.0716802441272244 | 0.05348506676105318 | 0.08987542149339563 |
| ciciot2023 | rare_f1_macro | 5 | 0.03348248659953164 | 0.021484854434861533 | 0.045480118764201743 |
| ciciot2023 | balanced_accuracy | 5 | 0.05607952255998257 | 0.04107108086831766 | 0.07108796425164748 |
| ciciot2023 | mcc | 5 | 0.06469669353547618 | 0.04788463938586134 | 0.081508747685091 |
| ciciot2023 | ece_15 | 5 | -0.009440957710767781 | -0.01936587719759353 | 0.0004839617760579659 |
| cse_cic_ids2018 | f1_macro | 5 | 0.02640047319536214 | 0.007938014797059006 | 0.04486293159366528 |
| cse_cic_ids2018 | rare_f1_macro | 5 | 0.015318868674075392 | -0.03296284608569008 | 0.06360058343384087 |
| cse_cic_ids2018 | balanced_accuracy | 5 | 0.00665192651118649 | -0.009059911149538634 | 0.02236376417191161 |
| cse_cic_ids2018 | mcc | 5 | 0.02512670278650133 | 0.014019231885181765 | 0.03623417368782089 |
| cse_cic_ids2018 | ece_15 | 5 | -0.045201984166309536 | -0.07677492035806495 | -0.013629047974554123 |
| edge_iiotset | f1_macro | 5 | 0.0675582016167439 | 0.044314172073799604 | 0.0908022311596882 |
| edge_iiotset | rare_f1_macro | 5 | 0.062218826924709314 | 0.011852102928611247 | 0.11258555092080738 |
| edge_iiotset | balanced_accuracy | 5 | 0.04897575082141614 | 0.031183485988260717 | 0.06676801565457156 |
| edge_iiotset | mcc | 5 | 0.09807908299671823 | 0.062339423863490215 | 0.13381874212994627 |
| edge_iiotset | ece_15 | 5 | -0.024719840705719927 | -0.033715244880518316 | -0.015724436530921537 |
| ton_iot | f1_macro | 5 | 0.023495548507304266 | 0.011094125386821129 | 0.035896971627787405 |
| ton_iot | rare_f1_macro | 0 | nan | nan | nan |
| ton_iot | balanced_accuracy | 5 | 0.02349999999999992 | 0.010759999999999966 | 0.036239999999999876 |
| ton_iot | mcc | 5 | 0.02600555876686217 | 0.011780624708737255 | 0.040230492824987085 |
| ton_iot | ece_15 | 5 | -0.032324925944209076 | -0.04425327000803676 | -0.02039658188038139 |

## Paired Statistical Tests

| dataset | metric | paired_seeds | mean_difference_rareguard_minus_nova | paired_cohen_dz | paired_sign_effect | wilcoxon_p | holm_adjusted_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cic_bccc_nrc_tabulariotattack_2024 | f1_macro | 5 | 0.03473034246310418 | 0.8548215934134504 | 1.0 | 0.0625 | 1.0 |
| cic_bccc_nrc_tabulariotattack_2024 | rare_f1_macro | 5 | 0.0017554984349524155 | 0.03460142342069008 | -0.2 | 0.8125 | 1.0 |
| cic_bccc_nrc_tabulariotattack_2024 | balanced_accuracy | 5 | 0.0067204954894817485 | 0.16504970395476676 | -0.2 | 1.0 | 1.0 |
| cic_bccc_nrc_tabulariotattack_2024 | mcc | 5 | 0.03326763564600414 | 0.8378543111847032 | 0.6 | 0.125 | 1.0 |
| cic_bccc_nrc_tabulariotattack_2024 | ece_15 | 5 | -0.016935328874393103 | -1.6992040669355615 | -1.0 | 0.0625 | 1.0 |
| cic_iiot_2025_datasense | f1_macro | 5 | 0.4761042216686536 | 13.544041458496313 | 1.0 | 0.0625 | 1.0 |
| cic_iiot_2025_datasense | rare_f1_macro | 5 | 0.4559791689330467 | 10.888671692430043 | 1.0 | 0.0625 | 1.0 |
| cic_iiot_2025_datasense | balanced_accuracy | 5 | 0.26155327904766656 | 6.741695841023375 | 1.0 | 0.0625 | 1.0 |
| cic_iiot_2025_datasense | mcc | 5 | 0.5513502447986631 | 14.465287372844346 | 1.0 | 0.0625 | 1.0 |
| cic_iiot_2025_datasense | ece_15 | 5 | -0.08864904634539897 | -3.2278719230498183 | -1.0 | 0.0625 | 1.0 |
| ciciot2023 | f1_macro | 5 | 0.0716802441272244 | 3.4531405191805944 | 1.0 | 0.0625 | 1.0 |
| ciciot2023 | rare_f1_macro | 5 | 0.03348248659953164 | 2.446207143656014 | 1.0 | 0.0625 | 1.0 |
| ciciot2023 | balanced_accuracy | 5 | 0.05607952255998257 | 3.275214699106436 | 1.0 | 0.0625 | 1.0 |
| ciciot2023 | mcc | 5 | 0.06469669353547618 | 3.373124528622769 | 1.0 | 0.0625 | 1.0 |
| ciciot2023 | ece_15 | 5 | -0.009440957710767781 | -0.833796617782527 | -0.6 | 0.1875 | 1.0 |
| cse_cic_ids2018 | f1_macro | 5 | 0.02640047319536214 | 1.2534102750747034 | 1.0 | 0.0625 | 1.0 |
| cse_cic_ids2018 | rare_f1_macro | 5 | 0.015318868674075392 | 0.278109020997126 | 0.2 | 0.625 | 1.0 |
| cse_cic_ids2018 | balanced_accuracy | 5 | 0.00665192651118649 | 0.37110049067211026 | -0.2 | 0.8125 | 1.0 |
| cse_cic_ids2018 | mcc | 5 | 0.02512670278650133 | 1.9828569675528231 | 1.0 | 0.0625 | 1.0 |
| cse_cic_ids2018 | ece_15 | 5 | -0.045201984166309536 | -1.2549129358875164 | -1.0 | 0.0625 | 1.0 |
| edge_iiotset | f1_macro | 5 | 0.0675582016167439 | 2.547638073753188 | 1.0 | 0.0625 | 1.0 |
| edge_iiotset | rare_f1_macro | 5 | 0.062218826924709314 | 1.0828023356438896 | 1.0 | 0.125 | 1.0 |
| edge_iiotset | balanced_accuracy | 5 | 0.04897575082141614 | 2.4127978518860345 | 1.0 | 0.0625 | 1.0 |
| edge_iiotset | mcc | 5 | 0.09807908299671823 | 2.4054540197520176 | 1.0 | 0.0625 | 1.0 |
| edge_iiotset | ece_15 | 5 | -0.024719840705719927 | -2.4087740038852044 | -1.0 | 0.0625 | 1.0 |
| ton_iot | f1_macro | 5 | 0.023495548507304266 | 1.6606768516209032 | 1.0 | 0.0625 | 1.0 |
| ton_iot | rare_f1_macro | 0 | nan | nan | 0.0 | nan | nan |
| ton_iot | balanced_accuracy | 5 | 0.02349999999999992 | 1.6168491529613864 | 1.0 | 0.0625 | 1.0 |
| ton_iot | mcc | 5 | 0.02600555876686217 | 1.6024592597421121 | 1.0 | 0.0625 | 1.0 |
| ton_iot | ece_15 | 5 | -0.032324925944209076 | -2.375354592868436 | -1.0 | 0.0625 | 1.0 |

## Evidence Boundary

Current experiments use bounded deterministic range samples across canonical Drive CSV files, not every source row. No full-corpus claim is permitted. No energy-in-joules claim is permitted without an external hardware power meter.

Smoke outputs under `artifacts/smoke` are pipeline evidence only and are excluded from publication claims.