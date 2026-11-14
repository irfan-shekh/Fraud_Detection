import json

with open('unseen_data_audit/data/full_unseen_audit_results.json') as f:
    d = json.load(f)

print("=== TEST A: ORIGINAL HOLDOUT ===")
for m in ['pre_transaction', 'post_transaction']:
    v = d['test_a_original_holdout'][m]
    print(f"{m}: Acc={v['accuracy']:.4f}, Prec={v['precision']:.4f}, Rec={v['recall']:.4f}, F1={v['f1_score']:.4f}, PR-AUC={v['pr_auc']:.4f}, ROC-AUC={v['roc_auc']:.4f}, TP={v['true_positives']}, FP={v['false_positives']}, FN={v['false_negatives']}, TN={v['true_negatives']}")

print("\n=== TEST B: INDEPENDENT UNSEEN ===")
for m in ['pre_transaction', 'post_transaction']:
    v = d['test_b_independent_unseen'][m]
    print(f"{m}: Acc={v['accuracy']:.4f}, Prec={v['precision']:.4f}, Rec={v['recall']:.4f}, F1={v['f1_score']:.4f}, PR-AUC={v['pr_auc']:.4f}, ROC-AUC={v['roc_auc']:.4f}, TP={v['true_positives']}, FP={v['false_positives']}, FN={v['false_negatives']}, TN={v['true_negatives']}")

print("\n=== TEST C: DISTRIBUTION SHIFTS ===")
for sc, sdata in d['test_c_distribution_shifts'].items():
    print(f"Scenario: {sc} (N={sdata['sample_count']}, Frauds={sdata['fraud_count']})")
    for m in ['pre_transaction', 'post_transaction']:
        v = sdata[m]
        print(f"  {m}: Prec={v['precision']:.4f}, Rec={v['recall']:.4f}, F1={v['f1_score']:.4f}, PR-AUC={v['pr_auc']:.4f}, ROC-AUC={v['roc_auc']:.4f}, TP={v['true_positives']}, FP={v['false_positives']}, FN={v['false_negatives']}")

print("\n=== TEST D: NOISE ROBUSTNESS ===")
for nlvl, sdata in d['test_d_noise_robustness'].items():
    print(f"Noise level: {nlvl}")
    for m in ['pre_transaction', 'post_transaction']:
        v = sdata[m]
        print(f"  {m}: Prec={v['precision']:.4f}, Rec={v['recall']:.4f}, F1={v['f1_score']:.4f}, PR-AUC={v['pr_auc']:.4f}, ROC-AUC={v['roc_auc']:.4f}, TP={v['true_positives']}, FP={v['false_positives']}, FN={v['false_negatives']}")

print("\n=== THRESHOLD SENSITIVITY (on Unseen Data) ===")
for th in ['0.30', '0.40', '0.50', '0.60', '0.70']:
    print(f"Threshold: {th}")
    for m in ['pre_transaction', 'post_transaction']:
        v = d['threshold_sensitivity'][th][m]
        print(f"  {m}: Prec={v['precision']:.4f}, Rec={v['recall']:.4f}, F1={v['f1_score']:.4f}, TP={v['true_positives']}, FP={v['false_positives']}, FN={v['false_negatives']}")

with open('unseen_data_audit/data/leakage_audit_summary.json') as f:
    lk = json.load(f)
print("\n=== LEAKAGE SUMMARY ===")
print(json.dumps(lk, indent=2))
