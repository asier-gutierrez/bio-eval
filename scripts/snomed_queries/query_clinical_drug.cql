MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'373873005'})
WHERE c1.FSN CONTAINS '(clinical drug)'
RETURN DISTINCT c1.sctid, c1.FSN;