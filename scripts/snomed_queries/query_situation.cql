MATCH p=(c1)-[r2:ISA*{active:'1'}]->(c2:ObjectConcept{sctid:'243796009'})
WHERE c1.FSN CONTAINS '(situation)'
RETURN DISTINCT c1.sctid, c1.FSN;

