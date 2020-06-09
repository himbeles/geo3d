def html_table_from_matrix(mat):
    return "<table><tr>{}</tr></table>".format(
        "</tr><tr>".join(
            "<td>{}</td>".format("</td><td>".join("{:1.8f}".format(_) for _ in row))
            for row in mat
        )
    )


def html_table_from_vector(vec, indices=None):
    if indices is None:
        return "<table><tr><td>{}</td></tr></table>".format(
            "</td></tr><tr><td>".join("{:1.5f}".format(_) for _ in vec)
        )
    else:
        return "<table><tr><td>{}</td></tr></table>".format(
            "</td></tr><tr><td>".join(
                "{}</td><td>{:1.5f}".format(_[0], _[1]) for _ in zip(indices, vec)
            )
        )
