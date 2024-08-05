CREATE TABLE brand_info (
    brand_info_id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_name TEXT NOT NULL,
    brand_code INTEGER NOT NULL,
    learned_model_name TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    create_by TEXT NOT NULL,
    update_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    update_by TEXT NOT NULL,
    is_valid INTEGER DEFAULT 1 NOT NULL
);

CREATE TABLE prediction_result (
    prediction_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    future_predictions TEXT DEFAULT '[]',
    days_list TEXT DEFAULT '[]',
    brand_code INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    create_by TEXT NOT NULL,
    update_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    update_by TEXT NOT NULL,
    is_valid INTEGER DEFAULT 1 NOT NULL
);

CREATE TABLE brand (
    brand_id INTEGER PRIMARY KEY AUTOINCREMENT,
    brand_name TEXT NOT NULL,
    brand_code INTEGER NOT NULL,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    create_by TEXT NOT NULL,
    update_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    update_by TEXT NOT NULL,
    is_valid INTEGER DEFAULT 1 NOT NULL
);

-- INSERT INTO brand_info (
--     brand_name, brand_code, learned_model_name, user_id, create_at, create_by, update_at, update_by, is_valid
-- ) VALUES (
--     'test', 1111, 'test.pth', 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 1
-- );

-- INSERT INTO prediction_result (
--     future_predictions, days_list, brand_code, user_id, create_at, create_by, update_at, update_by, is_valid
-- ) VALUES (
--     '[100.0,101.0,102.0]', '[2024-07-01,2024-07-02,2024-07-03]', 1111, 1, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 1
-- );

-- INSERT INTO brand (
--     brand_name, brand_code, create_at, create_by, update_at, update_by, is_valid
-- ) VALUES (
--     'テスト', 1111, datetime('now', 'localtime', '+9 hours'), 'TEST', datetime('now', 'localtime', '+9 hours'), 'TEST', 1
-- );
