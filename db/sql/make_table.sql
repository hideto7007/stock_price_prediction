-- データベースファイルを作成
PRAGMA foreign_keys = ON;

-- brand_info テーブルを作成
DROP TABLE IF EXISTS brand_info;
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

-- prediction_result テーブルを作成
DROP TABLE IF EXISTS prediction_result;
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

-- brand テーブルを作成
DROP TABLE IF EXISTS brand;
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

-- user テーブルを作成
DROP TABLE IF EXISTS user;
CREATE TABLE user (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    user_email TEXT NOT NULL UNIQUE,
    user_password TEXT NOT NULL UNIQUE,
    create_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    create_by TEXT NOT NULL,
    update_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    update_by TEXT NOT NULL,
    is_valid INTEGER DEFAULT 1 NOT NULL
);