MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'123037004'})
WHERE c1.FSN CONTAINS '(morphologic abnormality)'
RETURN DISTINCT c1.sctid, c1.FSN;


