alias tabml_build_protos='cd $TABML; protoc -I=./ --python_out=./ ./tabml/protos/*.proto;'
alias tabml_lint='cd $TABML; flake8 ./tabml ./tests ./examples; mypy ./tabml ./tests ./examples'
