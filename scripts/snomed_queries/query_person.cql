MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'48176007'})
WHERE c1.FSN CONTAINS '(person)'
RETURN DISTINCT c1.sctid, c1.FSN;

