MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'71388002'})
WHERE c1.FSN CONTAINS '(regime/therapy)'
RETURN DISTINCT c1.sctid, c1.FSN;