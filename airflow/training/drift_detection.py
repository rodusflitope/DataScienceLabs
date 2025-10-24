from scipy import stats

def detect_drift_in_transactions(reference_trans, new_trans, threshold=0.05):
    numeric_cols = ['size', 'num_deliver_per_week', 'num_visit_per_week']
    
    drift_detected = False
    
    for col in numeric_cols:
        _, p_value = stats.ks_2samp(reference_trans[col], new_trans[col])
        if p_value < threshold:
            drift_detected = True
            break
    
    return drift_detected
