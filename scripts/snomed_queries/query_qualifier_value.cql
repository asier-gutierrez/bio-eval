MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'362981000'})
WHERE c1.FSN CONTAINS '(qualifier value)'
RETURN DISTINCT c1.sctid, c1.FSN;

