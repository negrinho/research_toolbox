import numpy as np
import matplotlib.pyplot as plt

# perhaps add hlines if needed lines and stuff like that. this can be done.
# NOTE: perhaps can have some structured representation for the model.
# TODO: add the latex header. make it multi column too.
# TODO: perhaps a better way is to have a more object oriented way of adding
# data incrementally.
def generate_latex_table(mat, num_places, row_labels=None, column_labels=None,
        bold_type=None, filepath=None, show=True):
    assert bold_type == None or (bold_type[0] in {'smallest', 'largest'} and
        bold_type[1] in {'in_row', 'in_col', 'all'})
    assert row_labels == None or len(row_labels) == mat.shape[0] and (
        column_labels == None or len(column_labels) == mat.shape[1])

    # effective number of latex rows and cols
    num_rows = mat.shape[0]
    num_cols = mat.shape[1]
    if row_labels != None:
        num_rows += 1
    if column_labels != None:
        num_cols += 1

    # round data
    proc_mat = np.round(mat, num_places)

    # determine the bolded entries:
    bold_mat = np.zeros_like(mat, dtype='bool')
    if bold_type != None:
        if bold_type[0] == 'largest':
            aux_fn = np.argmax
        else:
            aux_fn = np.argmin

        # determine the bolded elements; if many conflict, bold them all.
        if bold_type[1] == 'in_row':
            idxs = aux_fn(proc_mat, axis=1)
            for i in xrange(mat.shape[0]):
                mval = proc_mat[i, idxs[i]]
                bold_mat[i, :] = (proc_mat[i, :] == mval)

        elif bold_type[1] == 'in_col':
            idxs = aux_fn(proc_mat, axis=0)
            for j in xrange(mat.shape[1]):
                mval = proc_mat[idxs[j], j]
                bold_mat[:, j] = (proc_mat[:, j] == mval)
        else:
            idx = aux_fn(proc_mat)
            for j in xrange(mat.shape[1]):
                mval = proc_mat[:][idx]
                bold_mat[:, j] = (proc_mat[:, j] == mval)

    # construct the strings for the data.
    data = np.zeros_like(mat, dtype=object)
    for (i, j) in tb_ut.iter_product(range(mat.shape[0]), range(mat.shape[1])):
        s = "%s" % proc_mat[i, j]
        if bold_mat[i, j]:
            s = "\\textbf{%s}" % s
        data[i, j] = s

    header = ''
    if column_labels != None:
        header = " & ".join(column_labels) + " \\\\ \n"
        if row_labels != None:
            header = "&" + header

    body = [" & ".join(data_row) + " \\\\ \n" for data_row in data]
    if row_labels != None:
        body = [lab + " & " + body_row for (lab, body_row) in zip(row_labels, body)]

    table = header + "".join(body)
    if filepath != None:
        with open(filepath, 'w') as f:
            f.write(table)
    if show:
        print table


# subplot example. needs to be refactored to something more general.
def show_grid(matrix_lst, subplot_num_rows, subplot_num_cols, title_lst=None,
        show_colorbar=True, grid_spacing=8):
    num_plots = len(matrix_lst)
    assert subplot_num_rows * subplot_num_cols >= num_plots

    fig = plt.figure()
    for idx, mat in enumerate(matrix_lst):
        num_rows, num_cols = mat.shape
        ax = fig.add_subplot(subplot_num_rows, subplot_num_cols, idx + 1)
        plot = ax.pcolor(mat.astype(np.float64), vmin=0.0, vmax=1.0)
        ax.set_aspect('equal')
        ax.set_xticks([grid_spacing * i for i in range(num_cols // grid_spacing)])
        ax.set_yticks([grid_spacing * i for i in range(num_rows // grid_spacing)])
        # ax.set_xticks(range(num_cols), minor=True)
        # ax.set_yticks(range(num_rows), minor=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.margins(x=0, y=0)
        # ax.set_xlim(left=0.0, right=num_cols)
        # ax.set_ylim(bottom=0.0, top=num_rows)
        ax.grid(which='both')
        if show_colorbar:
            fig.colorbar(plot)
        if title_lst is not None:
            ax.set_title(title_lst[idx])
    # maybe change to a small value.
    # fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
