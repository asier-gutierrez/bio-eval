MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'419891008'})
WHERE c1.FSN CONTAINS '(record artifact)'
RETURN DISTINCT c1.sctid, c1.FSN;

