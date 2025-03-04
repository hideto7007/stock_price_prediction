-- ステップ 1: 新しいテーブルを作成する
CREATE TABLE brand_info_new (
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

-- ステップ 2: 古いテーブルから新しいテーブルにデータをコピーする
INSERT INTO brand_info_new (
    brand_info_id, brand_name, brand_code, learned_model_name, user_id, create_at, create_by, update_at, update_by, is_valid
)
SELECT
    brand_code_id, brand_name, brand_code, learned_model_name, user_id, create_at, create_by, update_at, update_by, is_valid
FROM
    brand_info;

-- ステップ 3: 古いテーブルを削除する
DROP TABLE brand_info;

-- ステップ 4: 新しいテーブルの名前を元の名前に変更する
ALTER TABLE brand_info_new RENAME TO brand_info;
