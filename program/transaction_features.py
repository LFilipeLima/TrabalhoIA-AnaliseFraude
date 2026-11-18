"""Constrói uma linha de atributos usando somente o contexto da transação."""


class TransactionFeatureBuilder:
    def __init__(self, connection):
        self.connection = connection

    def transformar(self, transacao):
        """Retorna os 21 atributos, sem inserir ou modificar transações.

        Janelas temporais excluem todo o instante atual. Contagens acumuladas
        usam (tx_datetime, transaction_id), conforme o notebook da OBT.
        Reportes só entram quando max(data da compra, data do reporte) < t.
        """
        cursor = self.connection.cursor()
        try:
            return cursor.execute(
                self._QUERY,
                [transacao.transaction_id, transacao.tx_datetime,
                 transacao.tx_amount, transacao.card_id, transacao.terminal_id],
            ).df()
        finally:
            cursor.close()

    _QUERY = """
    WITH atual AS (
        SELECT ?::BIGINT AS transaction_id, ?::TIMESTAMP AS instante,
               ?::DOUBLE AS valor, ?::BIGINT AS cartao, ?::BIGINT AS terminal
    ),
    cartao AS (
        SELECT
            COUNT(*) FILTER (WHERE t.tx_datetime >= a.instante - INTERVAL '1 hour'
                AND t.tx_datetime < a.instante) AS card_tx_count_1h,
            COUNT(*) FILTER (WHERE t.tx_datetime >= a.instante - INTERVAL '24 hours'
                AND t.tx_datetime < a.instante) AS card_tx_count_24h,
            COUNT(DISTINCT t.terminal_id) FILTER (
                WHERE t.tx_datetime >= a.instante - INTERVAL '24 hours'
                AND t.tx_datetime < a.instante) AS card_distinct_terminals_24h,
            EXTRACT(EPOCH FROM (ANY_VALUE(a.instante) - MAX(t.tx_datetime)
                FILTER (WHERE t.tx_datetime < a.instante))) AS seconds_since_last_card_tx,
            COUNT(t.tx_amount) FILTER (WHERE t.tx_datetime >= a.instante - INTERVAL '30 days'
                AND t.tx_datetime < a.instante) AS card_history_count_30d,
            AVG(t.tx_amount) FILTER (WHERE t.tx_datetime >= a.instante - INTERVAL '30 days'
                AND t.tx_datetime < a.instante) AS card_amount_mean_30d,
            STDDEV_SAMP(t.tx_amount) FILTER (
                WHERE t.tx_datetime >= a.instante - INTERVAL '30 days'
                AND t.tx_datetime < a.instante) AS card_amount_std_30d,
            COUNT(*) FILTER (WHERE t.terminal_id = a.terminal
                AND t.tx_datetime < a.instante) AS usos_terminal,
            COUNT(*) AS anteriores,
            COUNT(DISTINCT t.terminal_id) AS card_distinct_terminals
        FROM transactions t CROSS JOIN atual a
        WHERE t.card_id = a.cartao AND (
            t.tx_datetime < a.instante OR
            (t.tx_datetime = a.instante AND t.transaction_id < a.transaction_id)
        )
    ),
    terminal AS (
        SELECT
            COUNT(*) FILTER (WHERE t.tx_datetime >= a.instante - INTERVAL '1 hour'
                AND t.tx_datetime < a.instante) AS terminal_tx_count_1h,
            COUNT(*) FILTER (WHERE t.tx_datetime >= a.instante - INTERVAL '24 hours'
                AND t.tx_datetime < a.instante) AS terminal_tx_count_24h,
            COUNT(DISTINCT t.card_id) FILTER (
                WHERE t.tx_datetime >= a.instante - INTERVAL '24 hours'
                AND t.tx_datetime < a.instante) AS terminal_distinct_cards_24h,
            EXTRACT(EPOCH FROM (ANY_VALUE(a.instante) - MAX(t.tx_datetime)
                FILTER (WHERE t.tx_datetime < a.instante))) AS seconds_since_last_terminal_tx,
            COUNT(*) AS anteriores
        FROM transactions t CROSS JOIN atual a
        WHERE t.terminal_id = a.terminal AND (
            t.tx_datetime < a.instante OR
            (t.tx_datetime = a.instante AND t.transaction_id < a.transaction_id)
        )
    ),
    reportes AS (
        SELECT
            COUNT(*) FILTER (WHERE t.card_id = a.cartao) AS fraudes_cartao,
            COUNT(*) FILTER (WHERE t.terminal_id = a.terminal) AS fraudes_terminal
        FROM transactions t CROSS JOIN atual a
        WHERE (t.card_id = a.cartao OR t.terminal_id = a.terminal)
          AND t.tx_fraud_report_date IS NOT NULL
          AND GREATEST(CAST(t.tx_datetime AS TIMESTAMP),
                       CAST(t.tx_fraud_report_date AS TIMESTAMP)) < a.instante
    )
    SELECT
        a.valor AS tx_amount,
        EXTRACT(HOUR FROM a.instante) AS hour,
        EXTRACT(DOW FROM a.instante) AS day_week,
        CASE WHEN a.instante >= TRY_CAST(s.terminal_operation_start AS TIMESTAMP)
            THEN FLOOR(EXTRACT(EPOCH FROM (
                a.instante - TRY_CAST(s.terminal_operation_start AS TIMESTAMP))) / 86400)
            ELSE NULL END AS terminal_age_days,
        r.fraudes_terminal::DOUBLE / NULLIF(t.anteriores, 0) AS terminal_fraud_rate,
        r.fraudes_cartao::DOUBLE / NULLIF(c.anteriores, 0) AS card_fraud_rate,
        c.card_distinct_terminals,
        CASE WHEN c.usos_terminal = 0 THEN 1 ELSE 0 END AS terminal_new_for_card,
        c.card_tx_count_1h, c.card_tx_count_24h, c.card_distinct_terminals_24h,
        c.seconds_since_last_card_tx,
        t.terminal_tx_count_1h, t.terminal_tx_count_24h, t.terminal_distinct_cards_24h,
        t.seconds_since_last_terminal_tx,
        c.card_history_count_30d, c.card_amount_mean_30d, c.card_amount_std_30d,
        CASE WHEN c.card_amount_std_30d > 0
            THEN (a.valor - c.card_amount_mean_30d) / c.card_amount_std_30d
            ELSE NULL END AS amount_z_score_30d,
        TRIM(REGEXP_REPLACE(s.terminal_soft_descriptor, '[0-9]+$', '')) AS terminal_category
    FROM atual a CROSS JOIN cartao c CROSS JOIN terminal t CROSS JOIN reportes r
    LEFT JOIN seller_terminals s ON s.terminal_id = a.terminal
    """
