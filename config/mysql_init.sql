CREATE DATABASE IF NOT EXISTS hf_exercise DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE hf_exercise;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    age INT,
    sex VARCHAR(16),
    nyha_class ENUM('I','II','III','IV'),
    comorbidities TEXT,
    surgery_history TEXT,
    exercise_goals TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS exercise_library (
    exercise_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    acsm_type ENUM('aerobic','resistance','balance','flexibility') NOT NULL,
    body_region VARCHAR(64) NOT NULL,
    difficulty_level ENUM('low','moderate','high') NOT NULL,
    allowed_conditions TEXT,
    contraindications TEXT,
    recommended_angles JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS weekly_plan (
    plan_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    day_of_week ENUM('Mon','Tue','Wed','Thu','Fri','Sat','Sun') NOT NULL,
    exercise_id INT NOT NULL,
    sets INT,
    reps_or_duration VARCHAR(32),
    intensity_level VARCHAR(32),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (exercise_id) REFERENCES exercise_library(exercise_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    plan_id INT,
    exercise_id INT,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (plan_id) REFERENCES weekly_plan(plan_id) ON DELETE SET NULL,
    FOREIGN KEY (exercise_id) REFERENCES exercise_library(exercise_id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS pose_time_series (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    timestamp DATETIME NOT NULL,
    joint_name VARCHAR(64) NOT NULL,
    angle_value DECIMAL(6,2) NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS pose_risk_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id INT NOT NULL,
    timestamp DATETIME NOT NULL,
    risk_level ENUM('low','medium','high') NOT NULL,
    max_deviation DECIMAL(6,2) NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS login_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    login_time DATETIME NOT NULL,
    device_info VARCHAR(255),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS usage_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    page_name VARCHAR(128) NOT NULL,
    action VARCHAR(128) NOT NULL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
