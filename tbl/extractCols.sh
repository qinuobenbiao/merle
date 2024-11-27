#!/bin/bash
set -e
cd "$(dirname "$0")"

echo TPC-H Part
python extractCols.py --inPath dbgen/part.tbl --outDir cols/tPart --useCols 3 6 --pad0
echo TPC-H Order
python extractCols.py --inPath dbgen/orders.tbl --outDir cols/tOrder --useCols 0 1 3 4 5 --needFill0
rm cols/tOrder/col0.bin
echo TPC-H Customer
python extractCols.py --inPath dbgen/customer.tbl --outDir cols/tCustomer --useCols 6 --pad0
echo TPC-H Lineitem
python extractCols.py --inPath dbgen/lineitem.tbl --outDir cols/tLineitem --useCols 0 1 4 6 10 12 14

echo SSB Customer
python extractCols.py --inPath ssb-dbgen/customer.tbl --outDir cols/sCustomer --useCols 3 5 --pad0
echo SSB Part
python extractCols.py --inPath ssb-dbgen/part.tbl --outDir cols/sPart --useCols 2 4 --pad0
echo SSB Supplier
python extractCols.py --inPath ssb-dbgen/supplier.tbl --outDir cols/sSupplier --useCols 3 5 --pad0
# echo SSB Date
# python extractCols.py --inPath ssb-dbgen/date.tbl --outDir cols/sDate --useCols 0 2 3 4 5 8 11
echo SSB Lineorder
python extractCols.py --inPath ssb-dbgen/lineorder.tbl --outDir cols/sLineorder --useCols 0 2 3 4 5 8 11
