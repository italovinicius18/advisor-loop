-- Placeholder q03 (TPC-DS-like) para a imagem spark-advisor. O controller
-- substitui/popula o SQL real das queries TPC-DS na fase de smoke ao vivo
-- (Task 9 Step 7); este placeholder so garante que o COPY queries/ do
-- Dockerfile funciona e que o driver tem algo executavel para validar o
-- pipeline pod->kubectl->resultado antes de trocar pelo SQL definitivo.
SELECT count(*)
FROM store_sales ss
JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
WHERE d.d_year = 2001
