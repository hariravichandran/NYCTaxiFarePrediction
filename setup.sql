// setup.sql
// there are more columns in AnimeList.csv but i am using a simplified dataset for now
// Data may need to be cleaned further before import
CREATE TABLE anime (
  anime_id INT PRIMARY KEY,
  title TEXT
);
