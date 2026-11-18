from io import StringIO
from ruamel.yaml import YAML

yaml_content = """
# 这是顶级注释
key1: value1  # 行尾注释
key2: value2

# 这是嵌套结构的注释
nested:
  subkey1: subvalue1
  subkey2: subvalue2  # 嵌套的行尾注释

# 列表的注释
list_key:
  - item1
  - item2  # 列表项的注释
"""


def test_ruamel_yaml_preserves_structure_and_comments():
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    data = yaml.load(yaml_content)
    assert data["key1"] == "value1"
    assert data["nested"]["subkey2"] == "subvalue2"

    data["key1"] = "new_value1"
    output = StringIO()
    yaml.dump(data, output)
    dumped = output.getvalue()
    assert "# 这是顶级注释" in dumped
    assert "new_value1" in dumped
