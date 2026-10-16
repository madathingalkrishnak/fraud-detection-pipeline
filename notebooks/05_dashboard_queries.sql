-- ============================================================
-- Fraud Detection Dashboard Queries
-- For use with Databricks SQL Analytics or Power BI DirectQuery
-- ============================================================

-- -----------------------------------------
-- 1. KPI Summary (Last 24 Hours)
-- -----------------------------------------
-- Use as a "Counter" visualization in the dashboard
SELECT
    COUNT(*)                                            AS total_transactions,
    SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) AS fraud_detected,
    ROUND(
        SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    )                                                   AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud_predicted = 1 THEN amount ELSE 0 END), 2) AS fraud_amount_total,
    ROUND(AVG(CASE WHEN is_fraud_predicted = 1 THEN amount END), 2)        AS avg_fraud_amount,
    ROUND(AVG(fraud_probability), 4)                    AS avg_fraud_score
FROM transactions_enriched
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 24 HOURS;


-- -----------------------------------------
-- 2. Fraud Rate Over Time (Hourly Trend)
-- -----------------------------------------
-- Visualization: Line chart
SELECT
    DATE_TRUNC('hour', timestamp)                       AS hour,
    COUNT(*)                                            AS total_txn,
    SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    )                                                   AS fraud_rate_pct
FROM transactions_enriched
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour;


-- -----------------------------------------
-- 3. Fraud by Merchant Category
-- -----------------------------------------
-- Visualization: Horizontal bar chart
SELECT
    merchant_category,
    COUNT(*)                                            AS total_txn,
    SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    )                                                   AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN is_fraud_predicted = 1 THEN amount END), 2) AS avg_fraud_amount
FROM transactions_enriched
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
GROUP BY merchant_category
ORDER BY fraud_count DESC;


-- -----------------------------------------
-- 4. Fraud by Entry Mode
-- -----------------------------------------
-- Visualization: Pie chart
SELECT
    entry_mode,
    COUNT(*)                                            AS total_txn,
    SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    )                                                   AS fraud_rate_pct
FROM transactions_enriched
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
GROUP BY entry_mode
ORDER BY fraud_count DESC;


-- -----------------------------------------
-- 5. Geographic Fraud Distribution
-- -----------------------------------------
-- Visualization: Map or bubble chart
SELECT
    country,
    city,
    COUNT(*)                                            AS total_txn,
    SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(
        SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    )                                                   AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud_predicted = 1 THEN amount ELSE 0 END), 2) AS fraud_total_amount
FROM transactions_enriched
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
GROUP BY country, city
HAVING fraud_count > 0
ORDER BY fraud_count DESC
LIMIT 50;


-- -----------------------------------------
-- 6. High-Risk Transactions (Real-Time Alert Feed)
-- -----------------------------------------
-- Visualization: Table with conditional formatting
SELECT
    transaction_id,
    timestamp,
    card_id,
    user_id,
    merchant_category,
    amount,
    country,
    entry_mode,
    ROUND(fraud_probability, 4)                         AS fraud_score,
    CASE
        WHEN fraud_probability >= 0.9 THEN 'CRITICAL'
        WHEN fraud_probability >= 0.7 THEN 'HIGH'
        WHEN fraud_probability >= 0.5 THEN 'MEDIUM'
        ELSE 'LOW'
    END                                                 AS risk_level
FROM transactions_enriched
WHERE is_fraud_predicted = 1
  AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 1 HOURS
ORDER BY fraud_probability DESC
LIMIT 100;


-- -----------------------------------------
-- 7. Model Performance (Confusion Matrix)
-- -----------------------------------------
-- Only works when actual labels are available (e.g., after investigation)
SELECT
    CASE
        WHEN is_fraud = 1 AND is_fraud_predicted = 1 THEN 'True Positive'
        WHEN is_fraud = 0 AND is_fraud_predicted = 1 THEN 'False Positive'
        WHEN is_fraud = 1 AND is_fraud_predicted = 0 THEN 'False Negative'
        WHEN is_fraud = 0 AND is_fraud_predicted = 0 THEN 'True Negative'
    END                                                 AS classification,
    COUNT(*)                                            AS count
FROM transactions_enriched
WHERE is_fraud IS NOT NULL
  AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
GROUP BY
    CASE
        WHEN is_fraud = 1 AND is_fraud_predicted = 1 THEN 'True Positive'
        WHEN is_fraud = 0 AND is_fraud_predicted = 1 THEN 'False Positive'
        WHEN is_fraud = 1 AND is_fraud_predicted = 0 THEN 'False Negative'
        WHEN is_fraud = 0 AND is_fraud_predicted = 0 THEN 'True Negative'
    END;


-- -----------------------------------------
-- 8. Fraud Amount Distribution
-- -----------------------------------------
-- Visualization: Histogram
SELECT
    CASE
        WHEN amount < 10    THEN '$0-$10'
        WHEN amount < 50    THEN '$10-$50'
        WHEN amount < 100   THEN '$50-$100'
        WHEN amount < 500   THEN '$100-$500'
        WHEN amount < 1000  THEN '$500-$1K'
        WHEN amount < 5000  THEN '$1K-$5K'
        ELSE '$5K+'
    END                                                 AS amount_range,
    COUNT(*)                                            AS txn_count,
    SUM(amount)                                         AS total_amount
FROM transactions_enriched
WHERE is_fraud_predicted = 1
  AND timestamp >= CURRENT_TIMESTAMP - INTERVAL 30 DAYS
GROUP BY
    CASE
        WHEN amount < 10    THEN '$0-$10'
        WHEN amount < 50    THEN '$10-$50'
        WHEN amount < 100   THEN '$50-$100'
        WHEN amount < 500   THEN '$100-$500'
        WHEN amount < 1000  THEN '$500-$1K'
        WHEN amount < 5000  THEN '$1K-$5K'
        ELSE '$5K+'
    END
ORDER BY MIN(amount);


-- -----------------------------------------
-- 9. Card-Level Risk Profile (Top Flagged Cards)
-- -----------------------------------------
SELECT
    card_id,
    COUNT(*)                                            AS total_txn,
    SUM(CASE WHEN is_fraud_predicted = 1 THEN 1 ELSE 0 END) AS fraud_flags,
    ROUND(AVG(fraud_probability), 4)                    AS avg_risk_score,
    ROUND(SUM(amount), 2)                               AS total_spend,
    ROUND(SUM(CASE WHEN is_fraud_predicted = 1 THEN amount ELSE 0 END), 2) AS flagged_amount,
    COUNT(DISTINCT country)                             AS unique_countries,
    COUNT(DISTINCT merchant_category)                   AS unique_categories
FROM transactions_enriched
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 7 DAYS
GROUP BY card_id
HAVING fraud_flags > 0
ORDER BY fraud_flags DESC, avg_risk_score DESC
LIMIT 50;


-- -----------------------------------------
-- 10. Pipeline Latency Monitoring
-- -----------------------------------------
-- Tracks the delay between transaction time and processing time
SELECT
    DATE_TRUNC('minute', processed_at)                  AS minute,
    COUNT(*)                                            AS records_processed,
    ROUND(AVG(
        UNIX_TIMESTAMP(processed_at) - UNIX_TIMESTAMP(timestamp)
    ), 2)                                               AS avg_latency_sec,
    ROUND(PERCENTILE(
        UNIX_TIMESTAMP(processed_at) - UNIX_TIMESTAMP(timestamp), 0.99
    ), 2)                                               AS p99_latency_sec
FROM transactions_enriched
WHERE processed_at >= CURRENT_TIMESTAMP - INTERVAL 1 HOURS
GROUP BY DATE_TRUNC('minute', processed_at)
ORDER BY minute;
