SELECT id, question, answer,
       1 - (embedding <=> %s::vector) AS similarity
FROM faq
ORDER BY embedding <=> %s::vector
LIMIT 5;
