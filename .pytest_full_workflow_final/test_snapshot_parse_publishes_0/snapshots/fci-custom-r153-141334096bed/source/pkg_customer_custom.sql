CREATE OR REPLACE PACKAGE BODY pkg_customer_custom AS
  c_country CONSTANT VARCHAR2(2) := 'MY';
  PROCEDURE update_customer IS BEGIN DBMS_OUTPUT.PUT_LINE(c_country); END;
END pkg_customer_custom;
/
