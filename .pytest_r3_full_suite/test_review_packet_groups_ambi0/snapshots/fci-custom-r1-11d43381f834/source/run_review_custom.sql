CREATE OR REPLACE PROCEDURE run_review_custom(p_sql VARCHAR2) IS
  l_id NUMBER;
BEGIN
  pkg_visible_custom.do_work();
  pkg_kernel.do_work();
  unknown_local();
  SELECT id INTO l_id FROM app.business_table;
  EXECUTE IMMEDIATE p_sql;
END;
/
