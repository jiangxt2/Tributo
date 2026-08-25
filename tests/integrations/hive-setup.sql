CREATE DATABASE IF NOT EXISTS tributo_it;

DROP TABLE IF EXISTS tributo_it.events;

CREATE TABLE tributo_it.events (
  id INT,
  category STRING,
  score DOUBLE
)
STORED AS ORC;

INSERT INTO tributo_it.events VALUES
  (1, 'drop', 0.2),
  (2, 'keep', 0.7),
  (3, 'keep', 0.9);
