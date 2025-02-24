INSERT INTO brand_info (
    brand_name,
    brand_code,
    learned_model_name,
    user_id,
    create_at,
    create_by,
    update_at,
    update_by,
    is_valid
) VALUES 
    ('日野自動車', 7205, '/stock_price_prediction/save/1/best_model_weight_brand_code_7205_seq_len_6.pth', 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 1),
    ('トヨタ自動車', 7203, '/stock_price_prediction/save/1/best_model_weight_brand_code_7203_seq_len_6.pth', 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 1),
    ('日本水産', 1332, '/stock_price_prediction/save/1/best_model_weight_brand_code_1332_seq_len_6.pth', 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 0);


INSERT INTO prediction_result (
    future_predictions,
    days_list,
    brand_code,
    user_id,
    create_at,
    create_by,
    update_at,
    update_by,
    is_valid
) VALUES
    (
        '[3275.2364538230318, 3236.6430789332394, 3183.342918009302, 3121.5807838391393, 3056.1028504116907, 2990.8326806156515, 2928.830116446982]',
        '["2024-07-05", "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11", "2024-07-12", "2024-07-16"]',
        7203, 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 1
    ),
    (
        '[3275.2364538230318, 3236.6430789332394, 3183.342918009302, 3121.5807838391393, 3056.1028504116907, 2990.8326806156515, 2928.830116446982]',
        '["2024-07-05", "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11", "2024-07-12", "2024-07-16"]',
        7202, 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 0
    ),
    (
        '[3275.2364538230318, 3236.6430789332394, 3183.342918009302, 3121.5807838391393, 3056.1028504116907, 2990.8326806156515, 2928.830116446982]',
        '["2024-07-05", "2024-07-08", "2024-07-09", "2024-07-10", "2024-07-11", "2024-07-12", "2024-07-16"]',
        7211, 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 1
    );


INSERT INTO brand (
    brand_name,
    brand_code,
    create_at,
    create_by,
    update_at,
    update_by,
    is_valid
) VALUES 
    ('日本水産', 1332, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('いすゞ自動車', 7202, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('トヨタ自動車', 7203, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('日野自動車', 7205, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('三菱自動車工業', 7211, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('マツダ', 7261, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('本田技研工業', 7267, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('アサヒグループホールディングス', 2502, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('キリンホールディングス', 2503, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1),
    ('スズキ', 7269, datetime('now', 'localtime', '+9 hours'), 'MASTER', datetime('now', 'localtime', '+9 hours'), 'MASTER', 1);


INSERT INTO user (user_name, user_email, user_password, create_by, update_by)
VALUES
    ('山田 太郎', 'taro.yamada@example.com', 'hashed_password_123', 'admin', 'admin');