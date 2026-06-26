import * as path from 'path';
import { workspace, ExtensionContext } from 'vscode';

import {
	LanguageClient,
	LanguageClientOptions,
	ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
	// Path to the vx-analyzer binary
	// Assuming vx-analyzer is compiled and available in target/debug
	const serverCommand = '/Users/adityak/go/Vx/target/debug/vx-analyzer';

	const serverOptions: ServerOptions = {
		run: { command: serverCommand },
		debug: { command: serverCommand }
	};

	const clientOptions: LanguageClientOptions = {
		documentSelector: [{ scheme: 'file', language: 'vx' }],
		synchronize: {
			fileEvents: workspace.createFileSystemWatcher('**/*.vx')
		}
	};

	client = new LanguageClient(
		'vxLanguageServer',
		'Vx Language Server',
		serverOptions,
		clientOptions
	);

	client.start().catch((err: any) => {
		import('vscode').then(vscode => {
			vscode.window.showErrorMessage(`Failed to start Vx Language Server: ${err.message}`);
		});
	});
}

export function deactivate(): Thenable<void> | undefined {
	if (!client) {
		return undefined;
	}
	return client.stop();
}
