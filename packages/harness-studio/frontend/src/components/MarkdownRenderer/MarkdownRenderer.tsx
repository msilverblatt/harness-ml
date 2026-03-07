import styles from './MarkdownRenderer.module.css';

interface MarkdownRendererProps {
    content: string;
}

function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function renderInline(text: string): string {
    let result = escapeHtml(text);
    // Bold
    result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    result = result.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Inline code
    result = result.replace(/`(.+?)`/g, '<code>$1</code>');
    return result;
}

function parseTable(lines: string[]): string {
    const rows = lines
        .filter(line => !line.match(/^\|[\s-:|]+\|$/))
        .map(line =>
            line
                .replace(/^\|/, '')
                .replace(/\|$/, '')
                .split('|')
                .map(cell => cell.trim())
        );

    if (rows.length === 0) return '';

    const [header, ...body] = rows;
    const headerHtml = header.map(cell => `<th>${renderInline(cell)}</th>`).join('');
    const bodyHtml = body
        .map(row => `<tr>${row.map(cell => `<td>${renderInline(cell)}</td>`).join('')}</tr>`)
        .join('');

    return `<table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
}

function parseContent(content: string): string {
    const lines = content.split('\n');
    const output: string[] = [];
    let i = 0;

    while (i < lines.length) {
        const line = lines[i];

        // Table detection: line starts with | and next line is separator
        if (line.startsWith('|') && i + 1 < lines.length && lines[i + 1].match(/^\|[\s-:|]+\|$/)) {
            const tableLines: string[] = [];
            while (i < lines.length && lines[i].startsWith('|')) {
                tableLines.push(lines[i]);
                i++;
            }
            output.push(parseTable(tableLines));
            continue;
        }

        // Headers
        const headerMatch = line.match(/^(#{1,3})\s+(.+)$/);
        if (headerMatch) {
            const level = headerMatch[1].length;
            output.push(`<h${level}>${renderInline(headerMatch[2])}</h${level}>`);
            i++;
            continue;
        }

        // Empty line
        if (line.trim() === '') {
            output.push('<br/>');
            i++;
            continue;
        }

        // Regular paragraph
        output.push(`<p>${renderInline(line)}</p>`);
        i++;
    }

    return output.join('');
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
    const html = parseContent(content);

    return (
        <div
            className={styles.markdown}
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
}
