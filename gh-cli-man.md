# CLI Documentation for `gh`

_Generated on Wed Nov 26 18:40:43 EST 2025_

---

## `gh`

Work seamlessly with GitHub from the command line.

USAGE
  gh <command> <subcommand> [flags]

CORE COMMANDS
  auth:		 Authenticate gh and git with GitHub
  browse:	 Open repositories, issues, pull requests, and more in the browser
  codespace:	 Connect to and manage codespaces
  gist:		 Manage gists
  issue:	 Manage issues
  org:		 Manage organizations
  pr:		 Manage pull requests
  project:	 Work with GitHub Projects.
  release:	 Manage releases
  repo:		 Manage repositories

GITHUB ACTIONS COMMANDS
  cache:	 Manage GitHub Actions caches
  run:		 View details about workflow runs
  workflow:	 View details about GitHub Actions workflows

ALIAS COMMANDS
  co:		 Alias for "pr checkout"

ADDITIONAL COMMANDS
  agent-task:	 Work with agent tasks (preview)
  alias:	 Create command shortcuts
  api:		 Make an authenticated GitHub API request
  attestation:	 Work with artifact attestations
  completion:	 Generate shell completion scripts
  config:	 Manage configuration for gh
  extension:	 Manage gh extensions
  gpg-key:	 Manage GPG keys
  label:	 Manage labels
  preview:	 Execute previews for gh features
  ruleset:	 View info about repo rulesets
  search:	 Search for repositories, issues, and pull requests
  secret:	 Manage GitHub secrets
  ssh-key:	 Manage SSH keys
  status:	 Print information about relevant issues, pull requests, and notifications across repositories
  variable:	 Manage GitHub Actions variables

HELP TOPICS
  accessibility: Learn about GitHub CLI's accessibility experiences
  actions:	 Learn about working with GitHub Actions
  environment:	 Environment variables that can be used with gh
  exit-codes:	 Exit codes used by gh
  formatting:	 Formatting options for JSON data exported from gh
  mintty:	 Information about using gh with MinTTY
  reference:	 A comprehensive reference of all gh commands

FLAGS
  --help      Show help for command
  --version   Show gh version

EXAMPLES
  $ gh issue create
  $ gh repo clone cli/cli
  $ gh pr checkout 321

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh agent-task`

Working with agent tasks in the GitHub CLI is in preview and
subject to change without notice.


USAGE
  gh agent-task <command> [flags]

ALIASES
  gh agent-tasks, gh agent, gh agents

AVAILABLE COMMANDS
  create:	 Create an agent task (preview)
  list:		 List agent tasks (preview)
  view:		 View an agent task session (preview)

INHERITED FLAGS
  --help   Show help for command

ARGUMENTS
  A task can be identified as argument in any of the following formats:
  - by pull request number, e.g. "123"; or
  - by session ID, e.g. "12345abc-12345-12345-12345-12345abc"; or
  - by URL, e.g. "https://github.com/OWNER/REPO/pull/123/agent-sessions/12345abc-12345-12345-12345-12345abc";

  Identifying tasks by pull request is not recommended for non-interactive use cases as
  there may be multiple tasks for a given pull request that require disambiguation.

EXAMPLES
  # List your most recent agent tasks
  $ gh agent-task list

  # Create a new agent task on the current repository
  $ gh agent-task create "Improve the performance of the data processing pipeline"

  # View details about agent tasks associated with a pull request
  $ gh agent-task view 123

  # View details about a specific agent task
  $ gh agent-task view 12345abc-12345-12345-12345-12345abc

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh agent-task create`

Create an agent task (preview)

USAGE
  gh agent-task create [<task description>] [flags]

FLAGS
  -b, --base string		 Base branch for the pull request (use default branch if not provided)
  -a, --custom-agent string	 Use a custom agent for the task. e.g., use 'my-agent' for the 'my-agent.md' agent
      --follow			 Follow agent session logs
  -F, --from-file file		 Read task description from file (use "-" to read from standard input)
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Create a task from an inline description
  $ gh agent-task create "build me a new app"

  # Create a task from an inline description and follow logs
  $ gh agent-task create "build me a new app" --follow

  # Create a task from a file
  $ gh agent-task create -F task-desc.md

  # Create a task with problem statement from stdin
  $ echo "build me a new app" | gh agent-task create -F -

  # Create a task with an editor
  $ gh agent-task create

  # Create a task with an editor and a file as a template
  $ gh agent-task create -F task-desc.md

  # Select a different base branch for the PR
  $ gh agent-task create "fix errors" --base branch

  # Create a task using the custom agent defined in '.github/agents/my-agent.md'
  $ gh agent-task create "build me a new app" --custom-agent my-agent

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh agent-task list`

List agent tasks (preview)

USAGE
  gh agent-task list [flags]

FLAGS
  -L, --limit int   Maximum number of agent tasks to fetch (default 30) (default 30)
  -w, --web	    Open agent tasks in the browser

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh agent-task view`

View an agent task session.


USAGE
  gh agent-task view [<session-id> | <pr-number> | <pr-url> | <pr-branch>] [flags]

FLAGS
      --follow			 Follow agent session logs
      --log			 Show agent session logs
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format
  -w, --web			 Open agent task in the browser

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # View an agent task by session ID
  $ gh agent-task view e2fa49d2-f164-4a56-ab99-498090b8fcdf

  # View an agent task by pull request number in current repo
  $ gh agent-task view 12345

  # View an agent task by pull request number
  $ gh agent-task view --repo OWNER/REPO 12345

  # View an agent task by pull request reference
  $ gh agent-task view OWNER/REPO#12345

  # View a pull request agents tasks in the browser
  $ gh agent-task view 12345 --web

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh alias`

Aliases can be used to make shortcuts for gh commands or to compose multiple commands.

Run `gh help alias set` to learn more.


USAGE
  gh alias <command> [flags]

AVAILABLE COMMANDS
  delete:	 Delete set aliases
  import:	 Import aliases from a YAML file
  list:		 List your aliases
  set:		 Create a shortcut for a gh command

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh alias delete`

Delete set aliases

USAGE
  gh alias delete {<alias> | --all} [flags]

FLAGS
  --all	  Delete all aliases

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh alias import`

Import aliases from the contents of a YAML file.

Aliases should be defined as a map in YAML, where the keys represent aliases and
the values represent the corresponding expansions. An example file should look like
the following:

    bugs: issue list --label=bug
    igrep: '!gh issue list --label="$1" | grep "$2"'
    features: |-
	issue list
	--label=enhancement

Use `-` to read aliases (in YAML format) from standard input.

The output from `gh alias list` can be used to produce a YAML file
containing your aliases, which you can use to import them from one machine to
another. Run `gh help alias list` to learn more.


USAGE
  gh alias import [<filename> | -] [flags]

FLAGS
  --clobber   Overwrite existing aliases of the same name

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Import aliases from a file
  $ gh alias import aliases.yml

  # Import aliases from standard input
  $ gh alias import -

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh alias list`

This command prints out all of the aliases gh is configured to use.


USAGE
  gh alias list [flags]

ALIASES
  gh alias ls

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh alias set`

Define a word that will expand to a full gh command when invoked.

The expansion may specify additional arguments and flags. If the expansion includes
positional placeholders such as `$1`, extra arguments that follow the alias will be
inserted appropriately. Otherwise, extra arguments will be appended to the expanded
command.

Use `-` as expansion argument to read the expansion string from standard input. This
is useful to avoid quoting issues when defining expansions.

If the expansion starts with `!` or if `--shell` was given, the expansion is a shell
expression that will be evaluated through the `sh` interpreter when the alias is
invoked. This allows for chaining multiple commands via piping and redirection.


USAGE
  gh alias set <alias> <expansion> [flags]

FLAGS
      --clobber	  Overwrite existing aliases of the same name
  -s, --shell	  Declare an alias to be passed through a shell interpreter

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Note: Command Prompt on Windows requires using double quotes for arguments
  $ gh alias set pv 'pr view'
  $ gh pv -w 123  #=> gh pr view -w 123

  $ gh alias set bugs 'issue list --label=bugs'
  $ gh bugs

  $ gh alias set homework 'issue list --assignee @me'
  $ gh homework

  $ gh alias set 'issue mine' 'issue list --mention @me'
  $ gh issue mine

  $ gh alias set epicsBy 'issue list --author="$1" --label="epic"'
  $ gh epicsBy vilmibm	#=> gh issue list --author="vilmibm" --label="epic"

  $ gh alias set --shell igrep 'gh issue list --label="$1" | grep "$2"'
  $ gh igrep epic foo  #=> gh issue list --label="epic" | grep "foo"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh api`

Makes an authenticated HTTP request to the GitHub API and prints the response.

The endpoint argument should either be a path of a GitHub API v3 endpoint, or
`graphql` to access the GitHub API v4.

Placeholder values `{owner}`, `{repo}`, and `{branch}` in the endpoint
argument will get replaced with values from the repository of the current
directory or the repository specified in the `GH_REPO` environment variable.
Note that in some shells, for example PowerShell, you may need to enclose
any value that contains `{...}` in quotes to prevent the shell from
applying special meaning to curly braces.

The `-p/--preview` flag enables opting into previews, which are feature-flagged,
experimental API endpoints or behaviors. The API expects opt-in via the `Accept`
header with format `application/vnd.github.<preview-name>-preview+json` and this
command facilitates that via `--preview <preview-name>`. To send a request for
the corsair and scarlet witch previews, you could use `-p corsair,scarlet-witch`
or `--preview corsair --preview scarlet-witch`.

The default HTTP request method is `GET` normally and `POST` if any parameters
were added. Override the method with `--method`.

Pass one or more `-f/--raw-field` values in `key=value` format to add static string
parameters to the request payload. To add non-string or placeholder-determined values, see
`-F/--field` below. Note that adding request parameters will automatically switch the
request method to `POST`. To send the parameters as a `GET` query string instead, use
`--method GET`.

The `-F/--field` flag has magic type conversion based on the format of the value:

- literal values `true`, `false`, `null`, and integer numbers get converted to
  appropriate JSON types;
- placeholder values `{owner}`, `{repo}`, and `{branch}` get populated with values
  from the repository of the current directory;
- if the value starts with `@`, the rest of the value is interpreted as a
  filename to read the value from. Pass `-` to read from standard input.

For GraphQL requests, all fields other than `query` and `operationName` are
interpreted as GraphQL variables.

To pass nested parameters in the request payload, use `key[subkey]=value` syntax when
declaring fields. To pass nested values as arrays, declare multiple fields with the
syntax `key[]=value1`, `key[]=value2`. To pass an empty array, use `key[]` without a
value.

To pass pre-constructed JSON or payloads in other formats, a request body may be read
from file specified by `--input`. Use `-` to read from standard input. When passing the
request body this way, any parameters specified via field flags are added to the query
string of the endpoint URL.

In `--paginate` mode, all pages of results will sequentially be requested until
there are no more pages of results. For GraphQL requests, this requires that the
original query accepts an `$endCursor: String` variable and that it fetches the
`pageInfo{ hasNextPage, endCursor }` set of fields from a collection. Each page is a separate
JSON array or object. Pass `--slurp` to wrap all pages of JSON arrays or objects
into an outer JSON array.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh api <endpoint> [flags]

FLAGS
      --cache duration	      Cache the response, e.g. "3600s", "60m", "1h"
  -F, --field key=value	      Add a typed parameter in key=value format (use "@<path>" or "@-" to read value from file or stdin)
  -H, --header key:value      Add a HTTP request header in key:value format
      --hostname string	      The GitHub hostname for the request (default "github.com")
  -i, --include		      Include HTTP response status line and headers in the output
      --input file	      The file to use as body for the HTTP request (use "-" to read from standard input)
  -q, --jq string	      Query to select values from the response using jq syntax
  -X, --method string	      The HTTP method for the request (default "GET")
      --paginate	      Make additional HTTP requests to fetch all pages of results
  -p, --preview strings	      Opt into GitHub API previews (names should omit '-preview')
  -f, --raw-field key=value   Add a string parameter in key=value format
      --silent		      Do not print the response body
      --slurp		      Use with "--paginate" to return an array of all pages of either JSON arrays or objects
  -t, --template string	      Format JSON output using a Go template; see "gh help formatting"
      --verbose		      Include full HTTP request and response in the output

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # List releases in the current repository
  $ gh api repos/{owner}/{repo}/releases

  # Post an issue comment
  $ gh api repos/{owner}/{repo}/issues/123/comments -f body='Hi from CLI'

  # Post nested parameter read from a file
  $ gh api gists -F 'files[myfile.txt][content]=@myfile.txt'

  # Add parameters to a GET request
  $ gh api -X GET search/issues -f q='repo:cli/cli is:open remote'

  # Use a JSON file as request body
  $ gh api repos/{owner}/{repo}/rulesets --input file.json

  # Set a custom HTTP header
  $ gh api -H 'Accept: application/vnd.github.v3.raw+json' ...

  # Opt into GitHub API previews
  $ gh api --preview baptiste,nebula ...

  # Print only specific fields from the response
  $ gh api repos/{owner}/{repo}/issues --jq '.[].title'

  # Use a template for the output
  $ gh api repos/{owner}/{repo}/issues --template \
    '{{range .}}{{.title}} ({{.labels | pluck "name" | join ", " | color "yellow"}}){{"\n"}}{{end}}'

  # Update allowed values of the "environment" custom property in a deeply nested array
  $ gh api -X PATCH /orgs/{org}/properties/schema \
     -F 'properties[][property_name]=environment' \
     -F 'properties[][default_value]=production' \
     -F 'properties[][allowed_values][]=staging' \
     -F 'properties[][allowed_values][]=production'

  # List releases with GraphQL
  $ gh api graphql -F owner='{owner}' -F name='{repo}' -f query='
    query($name: String!, $owner: String!) {
      repository(owner: $owner, name: $name) {
	releases(last: 3) {
	  nodes { tagName }
	}
      }
    }
  '

  # List all repositories for a user
  $ gh api graphql --paginate -f query='
    query($endCursor: String) {
      viewer {
	repositories(first: 100, after: $endCursor) {
	  nodes { nameWithOwner }
	  pageInfo {
	    hasNextPage
	    endCursor
	  }
	}
      }
    }
  '

  # Get the percentage of forks for the current user
  $ gh api graphql --paginate --slurp -f query='
    query($endCursor: String) {
      viewer {
	repositories(first: 100, after: $endCursor) {
	  nodes { isFork }
	  pageInfo {
	    hasNextPage
	    endCursor
	  }
	}
      }
    }
  ' | jq 'def count(e): reduce e as $_ (0;.+1);
  [.[].data.viewer.repositories.nodes[]] as $r | count(select($r[].isFork))/count($r[])'

ENVIRONMENT VARIABLES
  GH_TOKEN, GITHUB_TOKEN (in order of precedence): an authentication token for
  `github.com` API requests.

  GH_ENTERPRISE_TOKEN, GITHUB_ENTERPRISE_TOKEN (in order of precedence): an
  authentication token for API requests to GitHub Enterprise.

  GH_HOST: make the request to a GitHub host other than `github.com`.

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh attestation`

Download and verify artifact attestations.


USAGE
  gh attestation [subcommand] [flags]

ALIASES
  gh at

AVAILABLE COMMANDS
  download:	 Download an artifact's attestations for offline use
  trusted-root:	 Output trusted_root.jsonl contents, likely for offline verification
  verify:	 Verify an artifact's integrity using attestations

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh attestation download`

### NOTE: This feature is currently in public preview, and subject to change.

Download attestations associated with an artifact for offline use.

The command requires either:
* a file path to an artifact, or
* a container image URI (e.g. `oci://<image-uri>`)
  * (note that if you provide an OCI URL, you must already be authenticated with
its container registry)

In addition, the command requires either:
* the `--repo` flag (e.g. --repo github/example).
* the `--owner` flag (e.g. --owner github), or

The `--repo` flag value must match the name of the GitHub repository
that the artifact is linked with.

The `--owner` flag value must match the name of the GitHub organization
that the artifact's linked repository belongs to.

Any associated bundle(s) will be written to a file in the
current directory named after the artifact's digest. For example, if the
digest is "sha256:1234", the file will be named "sha256:1234.jsonl".

Colons are special characters on Windows and cannot be used in
file names. To accommodate, a dash will be used to separate the algorithm
from the digest in the attestations file name. For example, if the digest
is "sha256:1234", the file will be named "sha256-1234.jsonl".


USAGE
  gh attestation download [<file-path> | oci://<image-uri>] [--owner | --repo] [flags]

FLAGS
  -d, --digest-alg string	The algorithm used to compute a digest of the artifact: {sha256|sha512} (default "sha256")
      --hostname string		Configure host to use
  -L, --limit int		Maximum number of attestations to fetch (default 30)
  -o, --owner string		GitHub organization to scope attestation lookup by
      --predicate-type string	Filter attestations by provided predicate type
  -R, --repo string		Repository name in the format <owner>/<repo>

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Download attestations for a local artifact linked with an organization
  $ gh attestation download example.bin -o github

  # Download attestations for a local artifact linked with a repository
  $ gh attestation download example.bin -R github/example

  # Download attestations for an OCI image linked with an organization
  $ gh attestation download oci://example.com/foo/bar:latest -o github

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh attestation trusted-root`

Output contents for a trusted_root.jsonl file, likely for offline verification.

When using `gh attestation verify`, if your machine is on the internet,
this will happen automatically. But to do offline verification, you need to
supply a trusted root file with `--custom-trusted-root`; this command
will help you fetch a `trusted_root.jsonl` file for that purpose.

You can call this command without any flags to get a trusted root file covering
the Sigstore Public Good Instance as well as GitHub's Sigstore instance.

Otherwise you can use `--tuf-url` to specify the URL of a custom TUF
repository mirror, and `--tuf-root` should be the path to the
`root.json` file that you securely obtained out-of-band.

If you just want to verify the integrity of your local TUF repository, and don't
want the contents of a trusted_root.jsonl file, use `--verify-only`.


USAGE
  gh attestation trusted-root [--tuf-url <url> --tuf-root <file-path>] [--verify-only] [flags]

FLAGS
  --hostname string   Configure host to use
  --tuf-root string   Path to the TUF root.json file on disk
  --tuf-url string    URL to the TUF repository mirror
  --verify-only	      Don't output trusted_root.jsonl contents

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Get a trusted_root.jsonl for both Sigstore Public Good and GitHub's instance
  $ gh attestation trusted-root

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh attestation verify`

Verify the integrity and provenance of an artifact using its associated
cryptographically signed attestations.

## Understanding Verification

An attestation is a claim (i.e. a provenance statement) made by an actor
(i.e. a GitHub Actions workflow) regarding a subject (i.e. an artifact).

In order to verify an attestation, you must provide an artifact and validate:
* the identity of the actor that produced the attestation
* the expected attestation predicate type (the nature of the claim)

By default, this command enforces the `https://slsa.dev/provenance/v1`
predicate type. To verify other attestation predicate types use the
`--predicate-type` flag.

The "actor identity" consists of:
* the repository or the repository owner the artifact is linked with
* the Actions workflow that produced the attestation (a.k.a the
  signer workflow)

This identity is then validated against the attestation's certificate's
SourceRepository, SourceRepositoryOwner, and SubjectAlternativeName
(SAN) fields, among others.

It is up to you to decide how precisely you want to enforce this identity.

At a minimum, this command requires either:
* the `--owner` flag (e.g. --owner github), or
* the `--repo` flag (e.g. --repo github/example)

The more precisely you specify the identity, the more control you will
have over the security guarantees offered by the verification process.

Ideally, the path of the signer workflow is also validated using the
`--signer-workflow` or `--cert-identity` flags.

Please note: if your attestation was generated via a reusable workflow then
that reusable workflow is the signer whose identity needs to be validated.
In this situation, you must use either the `--signer-workflow` or
the `--signer-repo` flag.

For more options, see the other available flags.

## Loading Artifacts And Attestations

To specify the artifact, this command requires:
* a file path to an artifact, or
* a container image URI (e.g. `oci://<image-uri>`)
  * (note that if you provide an OCI URL, you must already be authenticated with
its container registry)

By default, this command will attempt to fetch relevant attestations via the
GitHub API using the values provided to `--owner` or  `--repo`.

To instead fetch attestations from your artifact's OCI registry, use the
`--bundle-from-oci` flag.

For offline verification using attestations stored on disk (c.f. the download command)
provide a path to the `--bundle` flag.

## Additional Policy Enforcement

Given the `--format=json` flag, upon successful verification this
command will output a JSON array containing one entry per verified attestation.

This output can then be used for additional policy enforcement, i.e. by being
piped into a policy engine.

Each object in the array contains two properties:
* an `attestation` object, which contains the bundle that was verified
* a `verificationResult` object, which is a parsed representation of the
  contents of the bundle that was verified.

Within the `verificationResult` object you will find:
* `signature.certificate`, which is a parsed representation of the X.509
  certificate embedded in the attestation,
* `verifiedTimestamps`, an array of objects denoting when the attestation
  was witnessed by a transparency log or a timestamp authority
* `statement`, which contains the `subject` array referencing artifacts,
  the `predicateType` field, and the `predicate` object which contains
  additional, often user-controllable, metadata

IMPORTANT: please note that only the `signature.certificate` and the
`verifiedTimestamps` properties contain values that cannot be
manipulated by the workflow that originated the attestation.

When dealing with attestations created within GitHub Actions, the contents of
`signature.certificate` are populated directly from the OpenID Connect
token that GitHub has generated. The contents of the `verifiedTimestamps`
array are populated from the signed timestamps originating from either a
transparency log or a timestamp authority – and likewise cannot be forged by users.

When designing policy enforcement using this output, special care must be taken
when examining the contents of the `statement.predicate` property:
should an attacker gain access to your workflow's execution context, they
could then falsify the contents of the `statement.predicate`.

To mitigate this attack vector, consider using a "trusted builder": when generating
an artifact, have the build and attestation signing occur within a reusable workflow
whose execution cannot be influenced by input provided through the caller workflow.

See above re: `--signer-workflow`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh attestation verify [<file-path> | oci://<image-uri>] [--owner | --repo] [flags]

FLAGS
  -b, --bundle string		     Path to bundle on disk, either a single bundle in a JSON file or a JSON lines file with multiple bundles
      --bundle-from-oci		     When verifying an OCI image, fetch the attestation bundle from the OCI registry instead of from GitHub
      --cert-identity string	     Enforce that the certificate's SubjectAlternativeName matches the provided value exactly
  -i, --cert-identity-regex string   Enforce that the certificate's SubjectAlternativeName matches the provided regex
      --cert-oidc-issuer string	     Enforce that the issuer of the OIDC token matches the provided value (default "https://token.actions.githubusercontent.com")
      --custom-trusted-root string   Path to a trusted_root.jsonl file; likely for offline verification
      --deny-self-hosted-runners     Fail verification for attestations generated on self-hosted runners
  -d, --digest-alg string	     The algorithm used to compute a digest of the artifact: {sha256|sha512} (default "sha256")
      --format string		     Output format: {json}
      --hostname string		     Configure host to use
  -q, --jq expression		     Filter JSON output using a jq expression
  -L, --limit int		     Maximum number of attestations to fetch (default 30)
      --no-public-good		     Do not verify attestations signed with Sigstore public good instance
  -o, --owner string		     GitHub organization to scope attestation lookup by
      --predicate-type string	     Enforce that verified attestations' predicate type matches the provided value (default "https://slsa.dev/provenance/v1")
  -R, --repo string		     Repository name in the format <owner>/<repo>
      --signer-digest string	     Enforce that the digest associated with the signer workflow matches the provided value
      --signer-repo string	     Enforce that the workflow that signed the attestation's repository matches the provided value (<owner>/<repo>)
      --signer-workflow string	     Enforce that the workflow that signed the attestation matches the provided value ([host/]<owner>/<repo>/<path>/<to>/<workflow>)
      --source-digest string	     Enforce that the digest associated with the source repository matches the provided value
      --source-ref string	     Enforce that the git ref associated with the source repository matches the provided value
  -t, --template string		     Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Verify an artifact linked with a repository
  $ gh attestation verify example.bin --repo github/example

  # Verify an artifact linked with an organization
  $ gh attestation verify example.bin --owner github

  # Verify an artifact and output the full verification result
  $ gh attestation verify example.bin --owner github --format json

  # Verify an OCI image using attestations stored on disk
  $ gh attestation verify oci://<image-uri> --owner github --bundle sha256:foo.jsonl

  # Verify an artifact signed with a reusable workflow
  $ gh attestation verify example.bin --owner github --signer-repo actions/example

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth`

Authenticate gh and git with GitHub

USAGE
  gh auth <command> [flags]

AVAILABLE COMMANDS
  login:	 Log in to a GitHub account
  logout:	 Log out of a GitHub account
  refresh:	 Refresh stored authentication credentials
  setup-git:	 Setup git with GitHub CLI
  status:	 Display active account and authentication state on each known GitHub host
  switch:	 Switch active GitHub account
  token:	 Print the authentication token gh uses for a hostname and account

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth login`

Authenticate with a GitHub host.

The default hostname is `github.com`. This can be overridden using the `--hostname`
flag.

The default authentication mode is a web-based browser flow. After completion, an
authentication token will be stored securely in the system credential store.
If a credential store is not found or there is an issue using it gh will fallback
to writing the token to a plain text file. See `gh auth status` for its
stored location.

Alternatively, use `--with-token` to pass in a personal access token (classic) on standard input.
The minimum required scopes for the token are: `repo`, `read:org`, and `gist`.
Take care when passing a fine-grained personal access token to `--with-token`
as the inherent scoping to certain resources may cause confusing behaviour when interacting with other
resources. Favour setting `GH_TOKEN` for fine-grained personal access token usage.

Alternatively, gh will use the authentication token found in environment variables.
This method is most suitable for "headless" use of gh such as in automation. See
`gh help environment` for more info.

To use gh in GitHub Actions, add `GH_TOKEN: ${{ github.token }}` to `env`.

The git protocol to use for git operations on this host can be set with `--git-protocol`,
or during the interactive prompting. Although login is for a single account on a host, setting
the git protocol will take effect for all users on the host.

Specifying `ssh` for the git protocol will detect existing SSH keys to upload,
prompting to create and upload a new key if one is not found. This can be skipped with
`--skip-ssh-key` flag.

For more information on OAuth scopes, see
<https://docs.github.com/en/developers/apps/building-oauth-apps/scopes-for-oauth-apps/>.


USAGE
  gh auth login [flags]

FLAGS
  -c, --clipboard	      Copy one-time OAuth device code to clipboard
  -p, --git-protocol string   The protocol to use for git operations on this host: {ssh|https}
  -h, --hostname string	      The hostname of the GitHub instance to authenticate with
      --insecure-storage      Save authentication credentials in plain text instead of credential store
  -s, --scopes strings	      Additional authentication scopes to request
      --skip-ssh-key	      Skip generate/upload SSH key prompt
  -w, --web		      Open a browser to authenticate
      --with-token	      Read token from standard input

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Start interactive setup
  $ gh auth login

  # Open a browser to authenticate and copy one-time OAuth code to clipboard
  $ gh auth login --web --clipboard

  # Authenticate against github.com by reading the token from a file
  $ gh auth login --with-token < mytoken.txt

  # Authenticate with specific host
  $ gh auth login --hostname enterprise.internal

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth logout`

Remove authentication for a GitHub account.

This command removes the stored authentication configuration
for an account. The authentication configuration is only
removed locally.

This command does not revoke authentication tokens.

To revoke all authentication tokens generated by the GitHub CLI:

1. Visit <https://github.com/settings/applications>
2. Select the "GitHub CLI" application
3. Select "Revoke Access"
4. Select "I understand, revoke access"

Note: this procedure will revoke all authentication tokens ever
generated by the GitHub CLI across all your devices.

For more information about revoking OAuth application tokens, see:
<https://docs.github.com/en/apps/oauth-apps/using-oauth-apps/reviewing-your-authorized-oauth-apps>


USAGE
  gh auth logout [flags]

FLAGS
  -h, --hostname string	  The hostname of the GitHub instance to log out of
  -u, --user string	  The account to log out of

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Select what host and account to log out of via a prompt
  $ gh auth logout

  # Log out of a specific host and specific account
  $ gh auth logout --hostname enterprise.internal --user monalisa

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth refresh`

Expand or fix the permission scopes for stored credentials for active account.

The `--scopes` flag accepts a comma separated list of scopes you want
your gh credentials to have. If no scopes are provided, the command
maintains previously added scopes.

The `--remove-scopes` flag accepts a comma separated list of scopes you
want to remove from your gh credentials. Scope removal is idempotent.
The minimum set of scopes (`repo`, `read:org`, and `gist`) cannot be removed.

The `--reset-scopes` flag resets the scopes for your gh credentials to
the default set of scopes for your auth flow.

If you have multiple accounts in `gh auth status` and want to refresh the credentials for an
inactive account, you will have to use `gh auth switch` to that account first before using
this command, and then switch back when you are done.

For more information on OAuth scopes, see
<https://docs.github.com/en/developers/apps/building-oauth-apps/scopes-for-oauth-apps/>.


USAGE
  gh auth refresh [flags]

FLAGS
  -c, --clipboard		Copy one-time OAuth device code to clipboard
  -h, --hostname string		The GitHub host to use for authentication
      --insecure-storage	Save authentication credentials in plain text instead of credential store
  -r, --remove-scopes strings	Authentication scopes to remove from gh
      --reset-scopes		Reset authentication scopes to the default minimum set of scopes
  -s, --scopes strings		Additional authentication scopes for gh to have

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Open a browser to add write:org and read:public_key scopes
  $ gh auth refresh --scopes write:org,read:public_key

  # Open a browser to ensure your authentication credentials have the correct minimum scopes
  $ gh auth refresh

  # Open a browser to idempotently remove the delete_repo scope
  $ gh auth refresh --remove-scopes delete_repo

  # Open a browser to re-authenticate with the default minimum scopes
  $ gh auth refresh --reset-scopes

  # Open a browser to re-authenticate and copy one-time OAuth code to clipboard
  $ gh auth refresh --clipboard

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth setup-git`

This command configures `git` to use GitHub CLI as a credential helper.
For more information on git credential helpers please reference:
<https://git-scm.com/docs/gitcredentials>.

By default, GitHub CLI will be set as the credential helper for all authenticated hosts.
If there is no authenticated hosts the command fails with an error.

Alternatively, use the `--hostname` flag to specify a single host to be configured.
If the host is not authenticated with, the command fails with an error.


USAGE
  gh auth setup-git [flags]

FLAGS
  -f, --force --hostname   Force setup even if the host is not known. Must be used in conjunction with --hostname
  -h, --hostname string	   The hostname to configure git for

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Configure git to use GitHub CLI as the credential helper for all authenticated hosts
  $ gh auth setup-git

  # Configure git to use GitHub CLI as the credential helper for enterprise.internal host
  $ gh auth setup-git --hostname enterprise.internal

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth status`

Display active account and authentication state on each known GitHub host.

For each host, the authentication state of each known account is tested and any issues are included in the output.
Each host section will indicate the active account, which will be used when targeting that host.

If an account on any host (or only the one given via `--hostname`) has authentication issues,
the command will exit with 1 and output to stderr. Note that when using the `--json` option, the command
will always exit with zero regardless of any authentication issues, unless there is a fatal error.

To change the active account for a host, see `gh auth switch`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh auth status [flags]

FLAGS
  -a, --active		  Display the active account only
  -h, --hostname string	  Check only a specific hostname's auth status
      --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --show-token	  Display the auth token
      --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  hosts

EXAMPLES
  # Display authentication status for all accounts on all hosts
  $ gh auth status

  # Display authentication status for the active account on a specific host
  $ gh auth status --active --hostname github.example.com

  # Display tokens in plain text
  $ gh auth status --show-token

  # Format authentication status as JSON
  $ gh auth status --json hosts

  # Include plain text token in JSON output
  $ gh auth status --json hosts --show-token

  # Format hosts as a flat JSON array
  $ gh auth status --json hosts --jq '.hosts | add'

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth switch`

Switch the active account for a GitHub host.

This command changes the authentication configuration that will
be used when running commands targeting the specified GitHub host.

If the specified host has two accounts, the active account will be switched
automatically. If there are more than two accounts, disambiguation will be
required either through the `--user` flag or an interactive prompt.

For a list of authenticated accounts you can run `gh auth status`.


USAGE
  gh auth switch [flags]

FLAGS
  -h, --hostname string	  The hostname of the GitHub instance to switch account for
  -u, --user string	  The account to switch to

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Select what host and account to switch to via a prompt
  $ gh auth switch

  # Switch the active account on a specific host to a specific user
  $ gh auth switch --hostname enterprise.internal --user monalisa

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh auth token`

This command outputs the authentication token for an account on a given GitHub host.

Without the `--hostname` flag, the default host is chosen.

Without the `--user` flag, the active account for the host is chosen.


USAGE
  gh auth token [flags]

FLAGS
  -h, --hostname string	  The hostname of the GitHub instance authenticated with
  -u, --user string	  The account to output the token for

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh browse`

Transition from the terminal to the web browser to view and interact with:

- Issues
- Pull requests
- Repository content
- Repository home page
- Repository settings


USAGE
  gh browse [<number> | <path> | <commit-sha>] [flags]

FLAGS
  -b, --branch string		 Select another branch by passing in the branch name
  -c, --commit string[="last"]	 Select another commit by passing in the commit SHA, default is the last commit
  -n, --no-browser		 Print destination URL instead of opening the browser
  -p, --projects		 Open repository projects
  -r, --releases		 Open repository releases
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format
  -s, --settings		 Open repository settings
  -w, --wiki			 Open repository wiki

INHERITED FLAGS
  --help   Show help for command

ARGUMENTS
  A browser location can be specified using arguments in the following format:
  - by number for issue or pull request, e.g. "123"; or
  - by path for opening folders and files, e.g. "cmd/gh/main.go"; or
  - by commit SHA

EXAMPLES
  # Open the home page of the current repository
  $ gh browse

  # Open the script directory of the current repository
  $ gh browse script/

  # Open issue or pull request 217
  $ gh browse 217

  # Open commit page
  $ gh browse 77507cd94ccafcf568f8560cfecde965fcfa63

  # Open repository settings
  $ gh browse --settings

  # Open main.go at line 312
  $ gh browse main.go:312

  # Open main.go with the repository at head of bug-fix branch
  $ gh browse main.go --branch bug-fix

  # Open main.go with the repository at commit 775007cd
  $ gh browse main.go --commit=77507cd94ccafcf568f8560cfecde965fcfa63

ENVIRONMENT VARIABLES
  To configure a web browser other than the default, use the BROWSER environment variable.

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh cache`

Work with GitHub Actions caches.

USAGE
  gh cache <command> [flags]

AVAILABLE COMMANDS
  delete:	 Delete GitHub Actions caches
  list:		 List GitHub Actions caches

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh cache list
  $ gh cache delete --all

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh cache delete`

Delete GitHub Actions caches.

Deletion requires authorization with the `repo` scope.


USAGE
  gh cache delete [<cache-id> | <cache-key> | --all] [flags]

FLAGS
  -a, --all			     Delete all caches
  -r, --ref string		     Delete by cache key and ref, formatted as refs/heads/<branch name> or refs/pull/<number>/merge
      --succeed-on-no-caches --all   Return exit code 0 if no caches found. Must be used in conjunction with --all

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Delete a cache by id
  $ gh cache delete 1234

  # Delete a cache by key
  $ gh cache delete cache-key

  # Delete a cache by id in a specific repo
  $ gh cache delete 1234 --repo cli/cli

  # Delete a cache by key and branch ref
  $ gh cache delete cache-key --ref refs/heads/feature-branch

  # Delete a cache by key and PR ref
  $ gh cache delete cache-key --ref refs/pull/<PR-number>/merge

  # Delete all caches (exit code 1 on no caches)
  $ gh cache delete --all

  # Delete all caches (exit code 0 on no caches)
  $ gh cache delete --all --succeed-on-no-caches

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh cache list`

List GitHub Actions caches

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh cache list [flags]

ALIASES
  gh cache ls

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -k, --key string	  Filter by cache key prefix
  -L, --limit int	  Maximum number of caches to fetch (default 30)
  -O, --order string	  Order of caches returned: {asc|desc} (default "desc")
  -r, --ref string	  Filter by ref, formatted as refs/heads/<branch name> or refs/pull/<number>/merge
  -S, --sort string	  Sort fetched caches: {created_at|last_accessed_at|size_in_bytes} (default "last_accessed_at")
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  createdAt, id, key, lastAccessedAt, ref, sizeInBytes, version

EXAMPLES
  # List caches for current repository
  $ gh cache list

  # List caches for specific repository
  $ gh cache list --repo cli/cli

  # List caches sorted by least recently accessed
  $ gh cache list --sort last_accessed_at --order asc

  # List caches that have keys matching a prefix (or that match exactly)
  $ gh cache list --key key-prefix

  # List caches for a specific branch, replace <branch-name> with the actual branch name
  $ gh cache list --ref refs/heads/<branch-name>

  # List caches for a specific pull request, replace <pr-number> with the actual pull request number
  $ gh cache list --ref refs/pull/<pr-number>/merge

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh co`

Check out a pull request in git

USAGE
  gh pr checkout [<number> | <url> | <branch>] [flags]

ALIASES
  gh pr co

FLAGS
  -b, --branch string	     Local branch name to use (default [the name of the head branch])
      --detach		     Checkout PR with a detached HEAD
  -f, --force		     Reset the existing local branch to the latest state of the pull request
      --recurse-submodules   Update all submodules after checkout

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Interactively select a PR from the 10 most recent to check out
  $ gh pr checkout

  # Checkout a specific PR
  $ gh pr checkout 32
  $ gh pr checkout https://github.com/OWNER/REPO/pull/32
  $ gh pr checkout feature

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace`

Connect to and manage codespaces

USAGE
  gh codespace [flags]

ALIASES
  gh cs

AVAILABLE COMMANDS
  code:		 Open a codespace in Visual Studio Code
  cp:		 Copy files between local and remote file systems
  create:	 Create a codespace
  delete:	 Delete codespaces
  edit:		 Edit a codespace
  jupyter:	 Open a codespace in JupyterLab
  list:		 List codespaces
  logs:		 Access codespace logs
  ports:	 List ports in a codespace
  rebuild:	 Rebuild a codespace
  ssh:		 SSH into a codespace
  stop:		 Stop a running codespace
  view:		 View details about a codespace

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace code`

Open a codespace in Visual Studio Code

USAGE
  gh codespace code [flags]

FLAGS
  -c, --codespace string    Name of the codespace
      --insiders	    Use the insiders version of Visual Studio Code
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)
  -w, --web		    Use the web version of Visual Studio Code

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace cp`

The `cp` command copies files between the local and remote file systems.

As with the UNIX `cp` command, the first argument specifies the source and the last
specifies the destination; additional sources may be specified after the first,
if the destination is a directory.

The `--recursive` flag is required if any source is a directory.

A `remote:` prefix on any file name argument indicates that it refers to
the file system of the remote (Codespace) machine. It is resolved relative
to the home directory of the remote user.

By default, remote file names are interpreted literally. With the `--expand` flag,
each such argument is treated in the manner of `scp`, as a Bash expression to
be evaluated on the remote machine, subject to expansion of tildes, braces, globs,
environment variables, and backticks. For security, do not use this flag with arguments
provided by untrusted users; see <https://lwn.net/Articles/835962/> for discussion.

By default, the `cp` command will create a public/private ssh key pair to authenticate with
the codespace inside the `~/.ssh directory`.


USAGE
  gh codespace cp [-e] [-r] [-- [<scp flags>...]] <sources>... <dest>

FLAGS
  -c, --codespace string    Name of the codespace
  -e, --expand		    Expand remote file names on remote shell
  -p, --profile string	    Name of the SSH profile to use
  -r, --recursive	    Recursively copy directories
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh codespace cp -e README.md 'remote:/workspaces/$RepositoryName/'
  $ gh codespace cp -e 'remote:~/*.go' ./gofiles/
  $ gh codespace cp -e 'remote:/workspaces/myproj/go.{mod,sum}' ./gofiles/
  $ gh codespace cp -e -- -F ~/.ssh/codespaces_config 'remote:~/*.go' ./gofiles/

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace create`

Create a codespace

USAGE
  gh codespace create [flags]

FLAGS
  -b, --branch string		    Repository branch
      --default-permissions	    Do not prompt to accept additional permissions requested by the codespace
      --devcontainer-path string    Path to the devcontainer.json file to use when creating codespace
  -d, --display-name string	    Display name for the codespace (48 characters or less)
      --idle-timeout duration	    Allowed inactivity before codespace is stopped, e.g. "10m", "1h"
  -l, --location string		    Location: {EastUs|SouthEastAsia|WestEurope|WestUs2} (determined automatically if not provided)
  -m, --machine string		    Hardware specifications for the VM
  -R, --repo string		    Repository name with owner: user/repo
      --retention-period duration   Allowed time after shutting down before the codespace is automatically deleted (maximum 30 days), e.g. "1h", "72h"
  -s, --status			    Show status of post-create command and dotfiles
  -w, --web			    Create codespace from browser, cannot be used with --display-name, --idle-timeout, or --retention-period

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace delete`

Delete codespaces based on selection criteria.

All codespaces for the authenticated user can be deleted, as well as codespaces for a
specific repository. Alternatively, only codespaces older than N days can be deleted.

Organization administrators may delete any codespace billed to the organization.


USAGE
  gh codespace delete [flags]

FLAGS
      --all		    Delete all codespaces
  -c, --codespace string    Name of the codespace
      --days N		    Delete codespaces older than N days
  -f, --force		    Skip confirmation for codespaces that contain unsaved changes
  -o, --org login	    The login handle of the organization (admin-only)
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)
  -u, --user username	    The username to delete codespaces for (used with --org)

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace edit`

Edit a codespace

USAGE
  gh codespace edit [flags]

FLAGS
  -c, --codespace string      Name of the codespace
  -d, --display-name string   Set the display name
  -m, --machine string	      Set hardware specifications for the VM
  -R, --repo string	      Filter codespace selection by repository name (user/repo)
      --repo-owner string     Filter codespace selection by repository owner (username or org)

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace jupyter`

Open a codespace in JupyterLab

USAGE
  gh codespace jupyter [flags]

FLAGS
  -c, --codespace string    Name of the codespace
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace list`

List codespaces of the authenticated user.

Alternatively, organization administrators may list all codespaces billed to the organization.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh codespace list [flags]

ALIASES
  gh codespace ls, gh cs ls

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -L, --limit int	  Maximum number of codespaces to list (default 30)
  -o, --org login	  The login handle of the organization to list codespaces for (admin-only)
  -R, --repo string	  Repository name with owner: user/repo
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -u, --user username	  The username to list codespaces for (used with --org)
  -w, --web		  List codespaces in the web browser, cannot be used with --user or --org

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  createdAt, displayName, gitStatus, lastUsedAt, machineName, name, owner,
  repository, state, vscsTarget

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace logs`

Access codespace logs

USAGE
  gh codespace logs [flags]

FLAGS
  -c, --codespace string    Name of the codespace
  -f, --follow		    Tail and follow the logs
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace ports`

List ports in a codespace

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh codespace ports [flags]

AVAILABLE COMMANDS
  forward:	 Forward ports
  visibility:	 Change the visibility of the forwarded port

FLAGS
  -c, --codespace string    Name of the codespace
  -q, --jq expression	    Filter JSON output using a jq expression
      --json fields	    Output JSON with the specified fields
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)
  -t, --template string	    Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  browseUrl, label, sourcePort, visibility

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace ports forward`

Forward ports

USAGE
  gh codespace ports forward <remote-port>:<local-port>... [flags]

INHERITED FLAGS
  -c, --codespace string    Name of the codespace
      --help		    Show help for command
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace ports visibility`

Change the visibility of the forwarded port

USAGE
  gh codespace ports visibility <port>:{public|private|org}... [flags]

INHERITED FLAGS
  -c, --codespace string    Name of the codespace
      --help		    Show help for command
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)

EXAMPLES
  $ gh codespace ports visibility 80:org 3000:private 8000:public

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace rebuild`

Rebuilding recreates your codespace.

Your code and any current changes will be preserved. Your codespace will be rebuilt using
your working directory's dev container. A full rebuild also removes cached Docker images.


USAGE
  gh codespace rebuild [flags]

FLAGS
  -c, --codespace string    Name of the codespace
      --full		    Perform a full rebuild
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace ssh`

The `ssh` command is used to SSH into a codespace. In its simplest form, you can
run `gh cs ssh`, select a codespace interactively, and connect.

The `ssh` command will automatically create a public/private ssh key pair in the
`~/.ssh` directory if you do not have an existing valid key pair. When selecting the
key pair to use, the preferred order is:

1. Key specified by `-i` in `<ssh-flags>`
2. Automatic key, if it already exists
3. First valid key pair in ssh config (according to `ssh -G`)
4. Automatic key, newly created

The `ssh` command also supports deeper integration with OpenSSH using a `--config`
option that generates per-codespace ssh configuration in OpenSSH format.
Including this configuration in your `~/.ssh/config` improves the user experience
of tools that integrate with OpenSSH, such as Bash/Zsh completion of ssh hostnames,
remote path completion for `scp/rsync/sshfs`, `git` ssh remotes, and so on.

Once that is set up (see the second example below), you can ssh to codespaces as
if they were ordinary remote hosts (using `ssh`, not `gh cs ssh`).

Note that the codespace you are connecting to must have an SSH server pre-installed.
If the docker image being used for the codespace does not have an SSH server,
install it in your `Dockerfile` or, for codespaces that use Debian-based images,
you can add the following to your `devcontainer.json`:

	"features": {
		"ghcr.io/devcontainers/features/sshd:1": {
			"version": "latest"
		}
	}


USAGE
  gh codespace ssh [<flags>...] [-- <ssh-flags>...] [<command>]

FLAGS
  -c, --codespace string    Name of the codespace
      --config		    Write OpenSSH configuration to stdout
  -d, --debug		    Log debug data to a file
      --debug-file string   Path of the file log to
      --profile string	    Name of the SSH profile to use
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)
      --server-port int	    SSH server port number (0 => pick unused)

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh codespace ssh

  $ gh codespace ssh --config > ~/.ssh/codespaces
  $ printf 'Match all\nInclude ~/.ssh/codespaces\n' >> ~/.ssh/config

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace stop`

Stop a running codespace

USAGE
  gh codespace stop [flags]

FLAGS
  -c, --codespace string    Name of the codespace
  -o, --org login	    The login handle of the organization (admin-only)
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)
  -u, --user username	    The username to stop codespace for (used with --org)

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh codespace view`

View details about a codespace

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh codespace view [flags]

FLAGS
  -c, --codespace string    Name of the codespace
  -q, --jq expression	    Filter JSON output using a jq expression
      --json fields	    Output JSON with the specified fields
  -R, --repo string	    Filter codespace selection by repository name (user/repo)
      --repo-owner string   Filter codespace selection by repository owner (username or org)
  -t, --template string	    Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  billableOwner, createdAt, devcontainerPath, displayName, environmentId,
  gitStatus, idleTimeoutMinutes, lastUsedAt, location, machineDisplayName,
  machineName, name, owner, prebuild, recentFolders, repository,
  retentionExpiresAt, retentionPeriodDays, state, vscsTarget

EXAMPLES
  # Select a codespace from a list of all codespaces you own
  $ gh cs view

  # View the details of a specific codespace
  $ gh cs view -c codespace-name-12345

  # View the list of all available fields for a codespace
  $ gh cs view --json

  # View specific fields for a codespace
  $ gh cs view --json displayName,machineDisplayName,state

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh config`

Display or change configuration settings for gh.

Current respected settings:
- `git_protocol`: the protocol to use for git clone and push operations `{https | ssh}` (default `https`)
- `editor`: the text editor program to use for authoring text
- `prompt`: toggle interactive prompting in the terminal `{enabled | disabled}` (default `enabled`)
- `prefer_editor_prompt`: toggle preference for editor-based interactive prompting in the terminal `{enabled | disabled}` (default `disabled`)
- `pager`: the terminal pager program to send standard output to
- `http_unix_socket`: the path to a Unix socket through which to make an HTTP connection
- `browser`: the web browser to use for opening URLs
- `color_labels`: whether to display labels using their RGB hex color codes in terminals that support truecolor `{enabled | disabled}` (default `disabled`)
- `accessible_colors`: whether customizable, 4-bit accessible colors should be used `{enabled | disabled}` (default `disabled`)
- `accessible_prompter`: whether an accessible prompter should be used `{enabled | disabled}` (default `disabled`)
- `spinner`: whether to use a animated spinner as a progress indicator `{enabled | disabled}` (default `enabled`)


USAGE
  gh config <command> [flags]

AVAILABLE COMMANDS
  clear-cache:	 Clear the cli cache
  get:		 Print the value of a given configuration key
  list:		 Print a list of configuration keys and values
  set:		 Update configuration with a value for the given key

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh config clear-cache`

Clear the cli cache

USAGE
  gh config clear-cache [flags]

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Clear the cli cache
  $ gh config clear-cache

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh config get`

Print the value of a given configuration key

USAGE
  gh config get <key> [flags]

FLAGS
  -h, --host string   Get per-host setting

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh config get git_protocol

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh config list`

Print a list of configuration keys and values

USAGE
  gh config list [flags]

ALIASES
  gh config ls

FLAGS
  -h, --host string   Get per-host configuration

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh config set`

Update configuration with a value for the given key

USAGE
  gh config set <key> <value> [flags]

FLAGS
  -h, --host string   Set per-host setting

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh config set editor vim
  $ gh config set editor "code --wait"
  $ gh config set git_protocol ssh --host github.com
  $ gh config set prompt disabled

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh extension`

GitHub CLI extensions are repositories that provide additional gh commands.

The name of the extension repository must start with `gh-` and it must contain an
executable of the same name. All arguments passed to the `gh <extname>` invocation
will be forwarded to the `gh-<extname>` executable of the extension.

An extension cannot override any of the core gh commands. If an extension name conflicts
with a core gh command, you can use `gh extension exec <extname>`.

When an extension is executed, gh will check for new versions once every 24 hours and display
an upgrade notice. See `gh help environment` for information on disabling extension notices.

For the list of available extensions, see <https://github.com/topics/gh-extension>.


USAGE
  gh extension [flags]

ALIASES
  gh extensions, gh ext

AVAILABLE COMMANDS
  browse:	 Enter a UI for browsing, adding, and removing extensions
  create:	 Create a new extension
  exec:		 Execute an installed extension
  install:	 Install a gh extension from a repository
  list:		 List installed extension commands
  remove:	 Remove an installed extension
  search:	 Search extensions to the GitHub CLI
  upgrade:	 Upgrade installed extensions

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh extension browse`

This command will take over your terminal and run a fully interactive
interface for browsing, adding, and removing gh extensions. A terminal
width greater than 100 columns is recommended.

To learn how to control this interface, press `?` after running to see
the help text.

Press `q` to quit.

Running this command with `--single-column` should make this command
more intelligible for users who rely on assistive technology like screen
readers or high zoom.

For a more traditional way to discover extensions, see:

	gh ext search

along with `gh ext install`, `gh ext remove`, and `gh repo view`.


USAGE
  gh extension browse [flags]

FLAGS
      --debug		Log to /tmp/extBrowse-*
  -s, --single-column	Render TUI with only one column of text

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh extension create`

Create a new extension

USAGE
  gh extension create [<name>] [flags]

FLAGS
  --precompiled string	 Create a precompiled extension. Possible values: go, other

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Use interactively
  $ gh extension create

  # Create a script-based extension
  $ gh extension create foobar

  # Create a Go extension
  $ gh extension create --precompiled=go foobar

  # Create a non-Go precompiled extension
  $ gh extension create --precompiled=other foobar

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh extension exec`

_Failed to get help for `gh extension exec`_

---

## `gh extension install`

Install a GitHub CLI extension from a GitHub or local repository.

For GitHub repositories, the repository argument can be specified in
`OWNER/REPO` format or as a full repository URL.
The URL format is useful when the repository is not hosted on `github.com`.

For remote repositories, the GitHub CLI first looks for the release artifacts assuming
that it's a binary extension i.e. prebuilt binaries provided as part of the release.
In the absence of a release, the repository itself is cloned assuming that it's a
script extension i.e. prebuilt executable or script exists on its root.

The `--pin` flag may be used to specify a tag or commit for binary and script
extensions respectively, the latest version is used otherwise.

For local repositories, often used while developing extensions, use `.` as the
value of the repository argument. Note the following:

- After installing an extension from a locally cloned repository, the GitHub CLI will
manage this extension as a symbolic link (or equivalent mechanism on Windows) pointing
to an executable file with the same name as the repository in the repository's root.
For example, if the repository is named `gh-foobar`, the symbolic link will point
to `gh-foobar` in the extension repository's root.
- When executing the extension, the GitHub CLI will run the executable file found
by following the symbolic link. If no executable file is found, the extension
will fail to execute.
- If the extension is precompiled, the executable file must be built manually and placed
in the repository's root.

For the list of available extensions, see <https://github.com/topics/gh-extension>.


USAGE
  gh extension install <repository> [flags]

FLAGS
  --force	 Force upgrade extension, or ignore if latest already installed
  --pin string	 Pin extension to a release tag or commit ref

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Install an extension from a remote repository hosted on GitHub
  $ gh extension install owner/gh-extension

  # Install an extension from a remote repository via full URL
  $ gh extension install https://my.ghes.com/owner/gh-extension

  # Install an extension from a local repository in the current working directory
  $ gh extension install .

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh extension remove`

Remove an installed extension

USAGE
  gh extension remove <name> [flags]

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh extension search`

Search for gh extensions.

With no arguments, this command prints out the first 30 extensions
available to install sorted by number of stars. More extensions can
be fetched by specifying a higher limit with the `--limit` flag.

When connected to a terminal, this command prints out three columns.
The first has a ✓ if the extension is already installed locally. The
second is the full name of the extension repository in `OWNER/REPO`
format. The third is the extension's description.

When not connected to a terminal, the ✓ character is rendered as the
word "installed" but otherwise the order and content of the columns
are the same.

This command behaves similarly to `gh search repos` but does not
support as many search qualifiers. For a finer grained search of
extensions, try using:

	gh search repos --topic "gh-extension"

and adding qualifiers as needed. See `gh help search repos` to learn
more about repository search.

For listing just the extensions that are already installed locally,
see:

	gh ext list

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh extension search [<query>] [flags]

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
      --license strings	  Filter based on license type
  -L, --limit int	  Maximum number of extensions to fetch (default 30)
      --order string	  Order of repositories returned, ignored unless '--sort' flag is specified: {asc|desc} (default "desc")
      --owner strings	  Filter on owner
      --sort string	  Sort fetched repositories: {forks|help-wanted-issues|stars|updated} (default "best-match")
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  Open the search query in the web browser

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  createdAt, defaultBranch, description, forksCount, fullName, hasDownloads,
  hasIssues, hasPages, hasProjects, hasWiki, homepage, id, isArchived, isDisabled,
  isFork, isPrivate, language, license, name, openIssuesCount, owner, pushedAt,
  size, stargazersCount, updatedAt, url, visibility, watchersCount

EXAMPLES
  # List the first 30 extensions sorted by star count, descending
  $ gh ext search

  # List more extensions
  $ gh ext search --limit 300

  # List extensions matching the term "branch"
  $ gh ext search branch

  # List extensions owned by organization "github"
  $ gh ext search --owner github

  # List extensions, sorting by recently updated, ascending
  $ gh ext search --sort updated --order asc

  # List extensions, filtering by license
  $ gh ext search --license MIT

  # Open search results in the browser
  $ gh ext search -w

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh extension upgrade`

Upgrade installed extensions

USAGE
  gh extension upgrade {<name> | --all} [flags]

FLAGS
  --all	      Upgrade all extensions
  --dry-run   Only display upgrades
  --force     Force upgrade extension

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist`

Work with GitHub gists.

USAGE
  gh gist <command> [flags]

AVAILABLE COMMANDS
  clone:	 Clone a gist locally
  create:	 Create a new gist
  delete:	 Delete a gist
  edit:		 Edit one of your gists
  list:		 List your gists
  rename:	 Rename a file in a gist
  view:		 View a gist

INHERITED FLAGS
  --help   Show help for command

ARGUMENTS
  A gist can be supplied as argument in either of the following formats:
  - by ID, e.g. 5b0e0062eb8e9654adad7bb1d81cc75f
  - by URL, e.g. "https://gist.github.com/OWNER/5b0e0062eb8e9654adad7bb1d81cc75f"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist clone`

Clone a GitHub gist locally.

A gist can be supplied as argument in either of the following formats:
- by ID, e.g. `5b0e0062eb8e9654adad7bb1d81cc75f`
- by URL, e.g. `https://gist.github.com/OWNER/5b0e0062eb8e9654adad7bb1d81cc75f`

Pass additional `git clone` flags by listing them after `--`.


USAGE
  gh gist clone <gist> [<directory>] [-- <gitflags>...]

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist create`

Create a new GitHub gist with given contents.

Gists can be created from one or multiple files. Alternatively, pass `-` as
filename to read from standard input.

By default, gists are secret; use `--public` to make publicly listed ones.


USAGE
  gh gist create [<filename>... | <pattern>... | -] [flags]

ALIASES
  gh gist new

FLAGS
  -d, --desc string	  A description for this gist
  -f, --filename string	  Provide a filename to be used when reading from standard input
  -p, --public		  List the gist publicly (default "secret")
  -w, --web		  Open the web browser with created gist

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Publish file 'hello.py' as a public gist
  $ gh gist create --public hello.py

  # Create a gist with a description
  $ gh gist create hello.py -d "my Hello-World program in Python"

  # Create a gist containing several files
  $ gh gist create hello.py world.py cool.txt

  # Create a gist containing several files using patterns
  $ gh gist create *.md *.txt artifact.*

  # Read from standard input to create a gist
  $ gh gist create -

  # Create a gist from output piped from another command
  $ cat cool.txt | gh gist create

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist delete`

Delete a GitHub gist.

To delete a gist interactively, use `gh gist delete` with no arguments.

To delete a gist non-interactively, supply the gist id or url.


USAGE
  gh gist delete {<id> | <url>} [flags]

FLAGS
  --yes	  Confirm deletion without prompting

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Delete a gist interactively
  $ gh gist delete

  # Delete a gist non-interactively
  $ gh gist delete 1234

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist edit`

Edit one of your gists

USAGE
  gh gist edit {<id> | <url>} [<filename>] [flags]

FLAGS
  -a, --add string	  Add a new file to the gist
  -d, --desc string	  New description for the gist
  -f, --filename string	  Select a file to edit
  -r, --remove string	  Remove a file from the gist

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist list`

List gists from your user account.

You can use a regular expression to filter the description, file names,
or even the content of files in the gist using `--filter`.

For supported regular expression syntax, see <https://pkg.go.dev/regexp/syntax>.

Use `--include-content` to include content of files, noting that
this will be slower and increase the rate limit used. Instead of printing a table,
code will be printed with highlights similar to `gh search code`:

	{{gist ID}} {{file name}}
	    {{description}}
		{{matching lines from content}}

No highlights or other color is printed when output is redirected.


USAGE
  gh gist list [flags]

ALIASES
  gh gist ls

FLAGS
      --filter expression   Filter gists using a regular expression
      --include-content	    Include gists' file content when filtering
  -L, --limit int	    Maximum number of gists to fetch (default 10)
      --public		    Show only public gists
      --secret		    Show only secret gists

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # List all secret gists from your user account
  $ gh gist list --secret

  # Find all gists from your user account mentioning "octo" anywhere
  $ gh gist list --filter octo --include-content

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist rename`

Rename a file in the given gist ID / URL.

USAGE
  gh gist rename {<id> | <url>} <old-filename> <new-filename> [flags]

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gist view`

View the given gist or select from recent gists.

USAGE
  gh gist view [<id> | <url>] [flags]

FLAGS
  -f, --filename string	  Display a single file from the gist
      --files		  List file names from the gist
  -r, --raw		  Print raw instead of rendered gist contents
  -w, --web		  Open gist in the browser

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gpg-key`

Manage GPG keys registered with your GitHub account.

USAGE
  gh gpg-key <command> [flags]

AVAILABLE COMMANDS
  add:		 Add a GPG key to your GitHub account
  delete:	 Delete a GPG key from your GitHub account
  list:		 Lists GPG keys in your GitHub account

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gpg-key add`

Add a GPG key to your GitHub account

USAGE
  gh gpg-key add [<key-file>] [flags]

FLAGS
  -t, --title string   Title for the new key

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gpg-key delete`

Delete a GPG key from your GitHub account

USAGE
  gh gpg-key delete <key-id> [flags]

FLAGS
  -y, --yes   Skip the confirmation prompt

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh gpg-key list`

Lists GPG keys in your GitHub account

USAGE
  gh gpg-key list [flags]

ALIASES
  gh gpg-key ls

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue`

Work with GitHub issues.

USAGE
  gh issue <command> [flags]

GENERAL COMMANDS
  create:	 Create a new issue
  list:		 List issues in a repository
  status:	 Show status of relevant issues

TARGETED COMMANDS
  close:	 Close issue
  comment:	 Add a comment to an issue
  delete:	 Delete issue
  develop:	 Manage linked branches for an issue
  edit:		 Edit issues
  lock:		 Lock issue conversation
  pin:		 Pin a issue
  reopen:	 Reopen issue
  transfer:	 Transfer issue to another repository
  unlock:	 Unlock issue conversation
  unpin:	 Unpin a issue
  view:		 View an issue

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

ARGUMENTS
  An issue can be supplied as argument in any of the following formats:
  - by number, e.g. "123"; or
  - by URL, e.g. "https://github.com/OWNER/REPO/issues/123".

EXAMPLES
  $ gh issue list
  $ gh issue create --label bug
  $ gh issue view 123 --web

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue close`

Close issue

USAGE
  gh issue close {<number> | <url>} [flags]

FLAGS
  -c, --comment string	 Leave a closing comment
  -r, --reason string	 Reason for closing: {completed|not planned}

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue comment`

Add a comment to a GitHub issue.

Without the body text supplied through flags, the command will interactively
prompt for the comment text.


USAGE
  gh issue comment {<number> | <url>} [flags]

FLAGS
  -b, --body text	 The comment body text
  -F, --body-file file	 Read body text from file (use "-" to read from standard input)
      --create-if-none	 Create a new comment if no comments are found. Can be used only with --edit-last
      --delete-last	 Delete the last comment of the current user
      --edit-last	 Edit the last comment of the current user
  -e, --editor		 Skip prompts and open the text editor to write the body in
  -w, --web		 Open the web browser to write the comment
      --yes		 Skip the delete confirmation prompt when --delete-last is provided

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  $ gh issue comment 12 --body "Hi from GitHub CLI"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue create`

Create an issue on GitHub.

Adding an issue to projects requires authorization with the `project` scope.
To authorize, run `gh auth refresh -s project`.

The `--assignee` flag supports the following special values:
- `@me`: assign yourself
- `@copilot`: assign Copilot (not supported on GitHub Enterprise Server)


USAGE
  gh issue create [flags]

ALIASES
  gh issue new

FLAGS
  -a, --assignee login	 Assign people by their login. Use "@me" to self-assign.
  -b, --body string	 Supply a body. Will prompt for one otherwise.
  -F, --body-file file	 Read body text from file (use "-" to read from standard input)
  -e, --editor		 Skip prompts and open the text editor to write the title and body in. The first line is the title and the remaining text is the body.
  -l, --label name	 Add labels by name
  -m, --milestone name	 Add the issue to a milestone by name
  -p, --project title	 Add the issue to projects by title
      --recover string	 Recover input from a failed run of create
  -T, --template name	 Template name to use as starting body text
  -t, --title string	 Supply a title. Will prompt for one otherwise.
  -w, --web		 Open the browser to create an issue

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  $ gh issue create --title "I found a bug" --body "Nothing works"
  $ gh issue create --label "bug,help wanted"
  $ gh issue create --label bug --label "help wanted"
  $ gh issue create --assignee monalisa,hubot
  $ gh issue create --assignee "@me"
  $ gh issue create --assignee "@copilot"
  $ gh issue create --project "Roadmap"
  $ gh issue create --template "Bug Report"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue delete`

Delete issue

USAGE
  gh issue delete {<number> | <url>} [flags]

FLAGS
  --yes	  Confirm deletion without prompting

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue develop`

Manage linked branches for an issue.

When using the `--base` flag, the new development branch will be created from the specified
remote branch. The new branch will be configured as the base branch for pull requests created using
`gh pr create`.


USAGE
  gh issue develop {<number> | <url>} [flags]

FLAGS
  -b, --base string	     Name of the remote branch you want to make your new branch from
      --branch-repo string   Name or URL of the repository where you want to create your new branch
  -c, --checkout	     Checkout the branch after creating it
  -l, --list		     List linked branches for the issue
  -n, --name string	     Name of the branch to create

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # List branches for issue 123
  $ gh issue develop --list 123

  # List branches for issue 123 in repo cli/cli
  $ gh issue develop --list --repo cli/cli 123

  # Create a branch for issue 123 based on the my-feature branch
  $ gh issue develop 123 --base my-feature

  # Create a branch for issue 123 and check it out
  $ gh issue develop 123 --checkout

  # Create a branch in repo monalisa/cli for issue 123 in repo cli/cli
  $ gh issue develop 123 --repo cli/cli --branch-repo monalisa/cli

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue edit`

Edit one or more issues within the same repository.

Editing issues' projects requires authorization with the `project` scope.
To authorize, run `gh auth refresh -s project`.

The `--add-assignee` and `--remove-assignee` flags both support
the following special values:
- `@me`: assign or unassign yourself
- `@copilot`: assign or unassign Copilot (not supported on GitHub Enterprise Server)


USAGE
  gh issue edit {<numbers> | <urls>} [flags]

FLAGS
      --add-assignee login	Add assigned users by their login. Use "@me" to assign yourself, or "@copilot" to assign Copilot.
      --add-label name		Add labels by name
      --add-project title	Add the issue to projects by title
  -b, --body string		Set the new body.
  -F, --body-file file		Read body text from file (use "-" to read from standard input)
  -m, --milestone name		Edit the milestone the issue belongs to by name
      --remove-assignee login	Remove assigned users by their login. Use "@me" to unassign yourself, or "@copilot" to unassign Copilot.
      --remove-label name	Remove labels by name
      --remove-milestone	Remove the milestone association from the issue
      --remove-project title	Remove the issue from projects by title
  -t, --title string		Set the new title.

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  $ gh issue edit 23 --title "I found a bug" --body "Nothing works"
  $ gh issue edit 23 --add-label "bug,help wanted" --remove-label "core"
  $ gh issue edit 23 --add-assignee "@me" --remove-assignee monalisa,hubot
  $ gh issue edit 23 --add-assignee "@copilot"
  $ gh issue edit 23 --add-project "Roadmap" --remove-project v1,v2
  $ gh issue edit 23 --milestone "Version 1"
  $ gh issue edit 23 --remove-milestone
  $ gh issue edit 23 --body-file body.txt
  $ gh issue edit 23 34 --add-label "help wanted"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue list`

List issues in a GitHub repository. By default, this only lists open issues.

The search query syntax is documented here:
<https://docs.github.com/en/search-github/searching-on-github/searching-issues-and-pull-requests>

On supported GitHub hosts, advanced issue search syntax can be used in the
`--search` query. For more information about advanced issue search, see:
<https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/filtering-and-searching-issues-and-pull-requests#building-advanced-filters-for-issues>

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh issue list [flags]

ALIASES
  gh issue ls

FLAGS
      --app string	   Filter by GitHub App author
  -a, --assignee string	   Filter by assignee
  -A, --author string	   Filter by author
  -q, --jq expression	   Filter JSON output using a jq expression
      --json fields	   Output JSON with the specified fields
  -l, --label strings	   Filter by label
  -L, --limit int	   Maximum number of issues to fetch (default 30)
      --mention string	   Filter by mention
  -m, --milestone string   Filter by milestone number or title
  -S, --search query	   Search issues with query
  -s, --state string	   Filter by state: {open|closed|all} (default "open")
  -t, --template string	   Format JSON output using a Go template; see "gh help formatting"
  -w, --web		   List issues in the web browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  assignees, author, body, closed, closedAt, closedByPullRequestsReferences,
  comments, createdAt, id, isPinned, labels, milestone, number, projectCards,
  projectItems, reactionGroups, state, stateReason, title, updatedAt, url

EXAMPLES
  $ gh issue list --label "bug" --label "help wanted"
  $ gh issue list --author monalisa
  $ gh issue list --assignee "@me"
  $ gh issue list --milestone "The big 1.0"
  $ gh issue list --search "error no:assignee sort:created-asc"
  $ gh issue list --state all

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue lock`

Lock issue conversation

USAGE
  gh issue lock {<number> | <url>} [flags]

FLAGS
  -r, --reason string	Optional reason for locking conversation (off_topic, resolved, spam, too_heated).

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue pin`

Pin an issue to a repository.

The issue can be specified by issue number or URL.


USAGE
  gh issue pin {<number> | <url>} [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Pin an issue to the current repository
  $ gh issue pin 23

  # Pin an issue by URL
  $ gh issue pin https://github.com/owner/repo/issues/23

  # Pin an issue to specific repository
  $ gh issue pin 23 --repo owner/repo

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue reopen`

Reopen issue

USAGE
  gh issue reopen {<number> | <url>} [flags]

FLAGS
  -c, --comment string	 Add a reopening comment

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue status`

Show status of relevant issues

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh issue status [flags]

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  assignees, author, body, closed, closedAt, closedByPullRequestsReferences,
  comments, createdAt, id, isPinned, labels, milestone, number, projectCards,
  projectItems, reactionGroups, state, stateReason, title, updatedAt, url

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue transfer`

Transfer issue to another repository

USAGE
  gh issue transfer {<number> | <url>} <destination-repo> [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue unlock`

Unlock issue conversation

USAGE
  gh issue unlock {<number> | <url>} [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue unpin`

Unpin an issue from a repository.

The issue can be specified by issue number or URL.


USAGE
  gh issue unpin {<number> | <url>} [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Unpin issue from the current repository
  $ gh issue unpin 23

  # Unpin issue by URL
  $ gh issue unpin https://github.com/owner/repo/issues/23

  # Unpin an issue from specific repository
  $ gh issue unpin 23 --repo owner/repo

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh issue view`

Display the title, body, and other information about an issue.

With `--web` flag, open the issue in a web browser instead.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh issue view {<number> | <url>} [flags]

FLAGS
  -c, --comments	  View issue comments
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  Open an issue in the browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  assignees, author, body, closed, closedAt, closedByPullRequestsReferences,
  comments, createdAt, id, isPinned, labels, milestone, number, projectCards,
  projectItems, reactionGroups, state, stateReason, title, updatedAt, url

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh label`

Work with GitHub labels.

USAGE
  gh label <command> [flags]

AVAILABLE COMMANDS
  clone:	 Clones labels from one repository to another
  create:	 Create a new label
  delete:	 Delete a label from a repository
  edit:		 Edit a label
  list:		 List labels in a repository

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh label clone`

Clones labels from a source repository to a destination repository on GitHub.
By default, the destination repository is the current repository.

All labels from the source repository will be copied to the destination
repository. Labels in the destination repository that are not in the source
repository will not be deleted or modified.

Labels from the source repository that already exist in the destination
repository will be skipped. You can overwrite existing labels in the
destination repository using the `--force` flag.


USAGE
  gh label clone <source-repository> [flags]

FLAGS
  -f, --force	Overwrite labels in the destination repository

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Clone and overwrite labels from cli/cli repository into the current repository
  $ gh label clone cli/cli --force

  # Clone labels from cli/cli repository into a octocat/cli repository
  $ gh label clone cli/cli --repo octocat/cli

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh label create`

Create a new label on GitHub, or update an existing one with `--force`.

Must specify name for the label. The description and color are optional.
If a color isn't provided, a random one will be chosen.

The label color needs to be 6 character hex value.


USAGE
  gh label create <name> [flags]

FLAGS
  -c, --color string	     Color of the label
  -d, --description string   Description of the label
  -f, --force		     Update the label color and description if label already exists

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Create new bug label
  $ gh label create bug --description "Something isn't working" --color E99695

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh label delete`

Delete a label from a repository

USAGE
  gh label delete <name> [flags]

FLAGS
  --yes	  Confirm deletion without prompting

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh label edit`

Update a label on GitHub.

A label can be renamed using the `--name` flag.

The label color needs to be 6 character hex value.


USAGE
  gh label edit <name> [flags]

FLAGS
  -c, --color string	     Color of the label
  -d, --description string   Description of the label
  -n, --name string	     New name of the label

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Update the color of the bug label
  $ gh label edit bug --color FF0000

  # Rename and edit the description of the bug label
  $ gh label edit bug --name big-bug --description "Bigger than normal bug"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh label list`

Display labels in a GitHub repository.

When using the `--search` flag results are sorted by best match of the query.
This behavior cannot be configured with the `--order` or `--sort` flags.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh label list [flags]

ALIASES
  gh label ls

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -L, --limit int	  Maximum number of labels to fetch (default 30)
      --order string	  Order of labels returned: {asc|desc} (default "asc")
  -S, --search string	  Search label names and descriptions
      --sort string	  Sort fetched labels: {created|name} (default "created")
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  List labels in the web browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  color, createdAt, description, id, isDefault, name, updatedAt, url

EXAMPLES
  # Sort labels by name
  $ gh label list --sort name

  # Find labels with "bug" in the name or description
  $ gh label list --search bug

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh org`

Work with GitHub organizations.

USAGE
  gh org <command> [flags]

GENERAL COMMANDS
  list:		 List organizations for the authenticated user.

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh org list

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh org list`

List organizations for the authenticated user.

USAGE
  gh org list [flags]

ALIASES
  gh org ls

FLAGS
  -L, --limit int   Maximum number of organizations to list (default 30)

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # List the first 30 organizations
  $ gh org list

  # List more organizations
  $ gh org list --limit 100

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr`

Work with GitHub pull requests.

USAGE
  gh pr <command> [flags]

GENERAL COMMANDS
  create:	 Create a pull request
  list:		 List pull requests in a repository
  status:	 Show status of relevant pull requests

TARGETED COMMANDS
  checkout:	 Check out a pull request in git
  checks:	 Show CI status for a single pull request
  close:	 Close a pull request
  comment:	 Add a comment to a pull request
  diff:		 View changes in a pull request
  edit:		 Edit a pull request
  lock:		 Lock pull request conversation
  merge:	 Merge a pull request
  ready:	 Mark a pull request as ready for review
  reopen:	 Reopen a pull request
  revert:	 Revert a pull request
  review:	 Add a review to a pull request
  unlock:	 Unlock pull request conversation
  update-branch: Update a pull request branch
  view:		 View a pull request

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

ARGUMENTS
  A pull request can be supplied as argument in any of the following formats:
  - by number, e.g. "123";
  - by URL, e.g. "https://github.com/OWNER/REPO/pull/123"; or
  - by the name of its head branch, e.g. "patch-1" or "OWNER:patch-1".

EXAMPLES
  $ gh pr checkout 353
  $ gh pr create --fill
  $ gh pr view --web

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr checkout`

Check out a pull request in git

USAGE
  gh pr checkout [<number> | <url> | <branch>] [flags]

ALIASES
  gh pr co

FLAGS
  -b, --branch string	     Local branch name to use (default [the name of the head branch])
      --detach		     Checkout PR with a detached HEAD
  -f, --force		     Reset the existing local branch to the latest state of the pull request
      --recurse-submodules   Update all submodules after checkout

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Interactively select a PR from the 10 most recent to check out
  $ gh pr checkout

  # Checkout a specific PR
  $ gh pr checkout 32
  $ gh pr checkout https://github.com/OWNER/REPO/pull/32
  $ gh pr checkout feature

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr checks`

Show CI status for a single pull request.

Without an argument, the pull request that belongs to the current branch
is selected.

When the `--json` flag is used, it includes a `bucket` field, which categorizes
the `state` field into `pass`, `fail`, `pending`, `skipping`, or `cancel`.

Additional exit codes:
	8: Checks pending

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh pr checks [<number> | <url> | <branch>] [flags]

FLAGS
      --fail-fast	  Exit watch mode on first check failure
  -i, --interval int	  Refresh interval in seconds in watch mode (default 10)
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
      --required	  Only show checks that are required
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
      --watch		  Watch checks until they finish
  -w, --web		  Open the web browser to show details about checks

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  bucket, completedAt, description, event, link, name, startedAt, state, workflow

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr close`

Close a pull request

USAGE
  gh pr close {<number> | <url> | <branch>} [flags]

FLAGS
  -c, --comment string	 Leave a closing comment
  -d, --delete-branch	 Delete the local and remote branch after close

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr comment`

Add a comment to a GitHub pull request.

Without the body text supplied through flags, the command will interactively
prompt for the comment text.


USAGE
  gh pr comment [<number> | <url> | <branch>] [flags]

FLAGS
  -b, --body text	 The comment body text
  -F, --body-file file	 Read body text from file (use "-" to read from standard input)
      --create-if-none	 Create a new comment if no comments are found. Can be used only with --edit-last
      --delete-last	 Delete the last comment of the current user
      --edit-last	 Edit the last comment of the current user
  -e, --editor		 Skip prompts and open the text editor to write the body in
  -w, --web		 Open the web browser to write the comment
      --yes		 Skip the delete confirmation prompt when --delete-last is provided

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  $ gh pr comment 13 --body "Hi from GitHub CLI"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr create`

Create a pull request on GitHub.

Upon success, the URL of the created pull request will be printed.

When the current branch isn't fully pushed to a git remote, a prompt will ask where
to push the branch and offer an option to fork the base repository. Use `--head` to
explicitly skip any forking or pushing behavior.

`--head` supports `<user>:<branch>` syntax to select a head repo owned by `<user>`.
Using an organization as the `<user>` is currently not supported.
For more information, see <https://github.com/cli/cli/issues/10093>

A prompt will also ask for the title and the body of the pull request. Use `--title` and
`--body` to skip this, or use `--fill` to autofill these values from git commits.
It's important to notice that if the `--title` and/or `--body` are also provided
alongside `--fill`, the values specified by `--title` and/or `--body` will
take precedence and overwrite any autofilled content.

The base branch for the created PR can be specified using the `--base` flag. If not provided,
the value of `gh-merge-base` git branch config will be used. If not configured, the repository's
default branch will be used. Run `git config branch.{current}.gh-merge-base {base}` to configure
the current branch to use the specified merge base.

Link an issue to the pull request by referencing the issue in the body of the pull
request. If the body text mentions `Fixes #123` or `Closes #123`, the referenced issue
will automatically get closed when the pull request gets merged.

By default, users with write access to the base repository can push new commits to the
head branch of the pull request. Disable this with `--no-maintainer-edit`.

Adding a pull request to projects requires authorization with the `project` scope.
To authorize, run `gh auth refresh -s project`.


USAGE
  gh pr create [flags]

ALIASES
  gh pr new

FLAGS
  -a, --assignee login	     Assign people by their login. Use "@me" to self-assign.
  -B, --base branch	     The branch into which you want your code merged
  -b, --body string	     Body for the pull request
  -F, --body-file file	     Read body text from file (use "-" to read from standard input)
  -d, --draft		     Mark pull request as a draft
      --dry-run		     Print details instead of creating the PR. May still push git changes.
  -e, --editor		     Skip prompts and open the text editor to write the title and body in. The first line is the title and the remaining text is the body.
  -f, --fill		     Use commit info for title and body
      --fill-first	     Use first commit info for title and body
      --fill-verbose	     Use commits msg+body for description
  -H, --head branch	     The branch that contains commits for your pull request (default [current branch])
  -l, --label name	     Add labels by name
  -m, --milestone name	     Add the pull request to a milestone by name
      --no-maintainer-edit   Disable maintainer's ability to modify pull request
  -p, --project title	     Add the pull request to projects by title
      --recover string	     Recover input from a failed run of create
  -r, --reviewer handle	     Request reviews from people or teams by their handle
  -T, --template file	     Template file to use as starting body text
  -t, --title string	     Title for the pull request
  -w, --web		     Open the web browser to create a pull request

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  $ gh pr create --title "The bug is fixed" --body "Everything works again"
  $ gh pr create --reviewer monalisa,hubot  --reviewer myorg/team-name
  $ gh pr create --project "Roadmap"
  $ gh pr create --base develop --head monalisa:feature
  $ gh pr create --template "pull_request_template.md"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr diff`

View changes in a pull request.

Without an argument, the pull request that belongs to the current branch
is selected.

With `--web` flag, open the pull request diff in a web browser instead.


USAGE
  gh pr diff [<number> | <url> | <branch>] [flags]

FLAGS
      --color string   Use color in diff output: {always|never|auto} (default "auto")
      --name-only      Display only names of changed files
      --patch	       Display diff in patch format
  -w, --web	       Open the pull request diff in the browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr edit`

Edit a pull request.

Without an argument, the pull request that belongs to the current branch
is selected.

Editing a pull request's projects requires authorization with the `project` scope.
To authorize, run `gh auth refresh -s project`.

The `--add-assignee` and `--remove-assignee` flags both support
the following special values:
- `@me`: assign or unassign yourself
- `@copilot`: assign or unassign Copilot (not supported on GitHub Enterprise Server)

The `--add-reviewer` and `--remove-reviewer` flags do not support
these special values.


USAGE
  gh pr edit [<number> | <url> | <branch>] [flags]

FLAGS
      --add-assignee login	Add assigned users by their login. Use "@me" to assign yourself, or "@copilot" to assign Copilot.
      --add-label name		Add labels by name
      --add-project title	Add the pull request to projects by title
      --add-reviewer login	Add reviewers by their login.
  -B, --base branch		Change the base branch for this pull request
  -b, --body string		Set the new body.
  -F, --body-file file		Read body text from file (use "-" to read from standard input)
  -m, --milestone name		Edit the milestone the pull request belongs to by name
      --remove-assignee login	Remove assigned users by their login. Use "@me" to unassign yourself, or "@copilot" to unassign Copilot.
      --remove-label name	Remove labels by name
      --remove-milestone	Remove the milestone association from the pull request
      --remove-project title	Remove the pull request from projects by title
      --remove-reviewer login	Remove reviewers by their login.
  -t, --title string		Set the new title.

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  $ gh pr edit 23 --title "I found a bug" --body "Nothing works"
  $ gh pr edit 23 --add-label "bug,help wanted" --remove-label "core"
  $ gh pr edit 23 --add-reviewer monalisa,hubot	 --remove-reviewer myorg/team-name
  $ gh pr edit 23 --add-assignee "@me" --remove-assignee monalisa,hubot
  $ gh pr edit 23 --add-assignee "@copilot"
  $ gh pr edit 23 --add-project "Roadmap" --remove-project v1,v2
  $ gh pr edit 23 --milestone "Version 1"
  $ gh pr edit 23 --remove-milestone

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr list`

List pull requests in a GitHub repository. By default, this only lists open PRs.

The search query syntax is documented here:
<https://docs.github.com/en/search-github/searching-on-github/searching-issues-and-pull-requests>

On supported GitHub hosts, advanced issue search syntax can be used in the
`--search` query. For more information about advanced issue search, see:
<https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/filtering-and-searching-issues-and-pull-requests#building-advanced-filters-for-issues>

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh pr list [flags]

ALIASES
  gh pr ls

FLAGS
      --app string	  Filter by GitHub App author
  -a, --assignee string	  Filter by assignee
  -A, --author string	  Filter by author
  -B, --base string	  Filter by base branch
  -d, --draft		  Filter by draft state
  -H, --head string	  Filter by head branch ("<owner>:<branch>" syntax not supported)
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -l, --label strings	  Filter by label
  -L, --limit int	  Maximum number of items to fetch (default 30)
  -S, --search query	  Search pull requests with query
  -s, --state string	  Filter by state: {open|closed|merged|all} (default "open")
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  List pull requests in the web browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  additions, assignees, author, autoMergeRequest, baseRefName, baseRefOid, body,
  changedFiles, closed, closedAt, closingIssuesReferences, comments, commits,
  createdAt, deletions, files, fullDatabaseId, headRefName, headRefOid,
  headRepository, headRepositoryOwner, id, isCrossRepository, isDraft, labels,
  latestReviews, maintainerCanModify, mergeCommit, mergeStateStatus, mergeable,
  mergedAt, mergedBy, milestone, number, potentialMergeCommit, projectCards,
  projectItems, reactionGroups, reviewDecision, reviewRequests, reviews, state,
  statusCheckRollup, title, updatedAt, url

EXAMPLES
  # List PRs authored by you
  $ gh pr list --author "@me"

  # List PRs with a specific head branch name
  $ gh pr list --head "typo"

  # List only PRs with all of the given labels
  $ gh pr list --label bug --label "priority 1"

  # Filter PRs using search syntax
  $ gh pr list --search "status:success review:required"

  # Find a PR that introduced a given commit
  $ gh pr list --search "<SHA>" --state merged

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr lock`

Lock pull request conversation

USAGE
  gh pr lock {<number> | <url>} [flags]

FLAGS
  -r, --reason string	Optional reason for locking conversation (off_topic, resolved, spam, too_heated).

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr merge`

Merge a pull request on GitHub.

Without an argument, the pull request that belongs to the current branch
is selected.

When targeting a branch that requires a merge queue, no merge strategy is required.
If required checks have not yet passed, auto-merge will be enabled.
If required checks have passed, the pull request will be added to the merge queue.
To bypass a merge queue and merge directly, pass the `--admin` flag.


USAGE
  gh pr merge [<number> | <url> | <branch>] [flags]

FLAGS
      --admin			Use administrator privileges to merge a pull request that does not meet requirements
  -A, --author-email text	Email text for merge commit author
      --auto			Automatically merge only after necessary requirements are met
  -b, --body text		Body text for the merge commit
  -F, --body-file file		Read body text from file (use "-" to read from standard input)
  -d, --delete-branch		Delete the local and remote branch after merge
      --disable-auto		Disable auto-merge for this pull request
      --match-head-commit SHA	Commit SHA that the pull request head must match to allow merge
  -m, --merge			Merge the commits with the base branch
  -r, --rebase			Rebase the commits onto the base branch
  -s, --squash			Squash the commits into one commit and merge it into the base branch
  -t, --subject text		Subject text for the merge commit

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr ready`

Mark a pull request as ready for review.

Without an argument, the pull request that belongs to the current branch
is marked as ready.

If supported by your plan, convert to draft with `--undo`


USAGE
  gh pr ready [<number> | <url> | <branch>] [flags]

FLAGS
  --undo   Convert a pull request to "draft"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr reopen`

Reopen a pull request

USAGE
  gh pr reopen {<number> | <url> | <branch>} [flags]

FLAGS
  -c, --comment string	 Add a reopening comment

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr revert`

Revert a pull request

USAGE
  gh pr revert {<number> | <url> | <branch>} [flags]

FLAGS
  -b, --body string	 Body for the revert pull request
  -F, --body-file file	 Read body text from file (use "-" to read from standard input)
  -d, --draft		 Mark revert pull request as a draft
  -t, --title string	 Title for the revert pull request

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr review`

Add a review to a pull request.

Without an argument, the pull request that belongs to the current branch is reviewed.


USAGE
  gh pr review [<number> | <url> | <branch>] [flags]

FLAGS
  -a, --approve		  Approve pull request
  -b, --body string	  Specify the body of a review
  -F, --body-file file	  Read body text from file (use "-" to read from standard input)
  -c, --comment		  Comment on a pull request
  -r, --request-changes	  Request changes on a pull request

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Approve the pull request of the current branch
  $ gh pr review --approve

  # Leave a review comment for the current branch
  $ gh pr review --comment -b "interesting"

  # Add a review for a specific pull request
  $ gh pr review 123

  # Request changes on a specific pull request
  $ gh pr review 123 -r -b "needs more ASCII art"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr status`

Show status of relevant pull requests.

The status shows a summary of pull requests that includes information such as
pull request number, title, CI checks, reviews, etc.

To see more details of CI checks, run `gh pr checks`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh pr status [flags]

FLAGS
  -c, --conflict-status	  Display the merge conflict status of each pull request
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  additions, assignees, author, autoMergeRequest, baseRefName, baseRefOid, body,
  changedFiles, closed, closedAt, closingIssuesReferences, comments, commits,
  createdAt, deletions, files, fullDatabaseId, headRefName, headRefOid,
  headRepository, headRepositoryOwner, id, isCrossRepository, isDraft, labels,
  latestReviews, maintainerCanModify, mergeCommit, mergeStateStatus, mergeable,
  mergedAt, mergedBy, milestone, number, potentialMergeCommit, projectCards,
  projectItems, reactionGroups, reviewDecision, reviewRequests, reviews, state,
  statusCheckRollup, title, updatedAt, url

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr unlock`

Unlock pull request conversation

USAGE
  gh pr unlock {<number> | <url>} [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr update-branch`

Update a pull request branch with latest changes of the base branch.

Without an argument, the pull request that belongs to the current branch is selected.

The default behavior is to update with a merge commit (i.e., merging the base branch
into the PR's branch). To reconcile the changes with rebasing on top of the base
branch, the `--rebase` option should be provided.


USAGE
  gh pr update-branch [<number> | <url> | <branch>] [flags]

FLAGS
  --rebase   Update PR branch by rebasing on top of latest base branch

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  $ gh pr update-branch 23
  $ gh pr update-branch 23 --rebase
  $ gh pr update-branch 23 --repo owner/repo

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh pr view`

Display the title, body, and other information about a pull request.

Without an argument, the pull request that belongs to the current branch
is displayed.

With `--web` flag, open the pull request in a web browser instead.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh pr view [<number> | <url> | <branch>] [flags]

FLAGS
  -c, --comments	  View pull request comments
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  Open a pull request in the browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  additions, assignees, author, autoMergeRequest, baseRefName, baseRefOid, body,
  changedFiles, closed, closedAt, closingIssuesReferences, comments, commits,
  createdAt, deletions, files, fullDatabaseId, headRefName, headRefOid,
  headRepository, headRepositoryOwner, id, isCrossRepository, isDraft, labels,
  latestReviews, maintainerCanModify, mergeCommit, mergeStateStatus, mergeable,
  mergedAt, mergedBy, milestone, number, potentialMergeCommit, projectCards,
  projectItems, reactionGroups, reviewDecision, reviewRequests, reviews, state,
  statusCheckRollup, title, updatedAt, url

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh preview`

Preview commands are for testing, demonstrative, and development purposes only.
They should be considered unstable and can change at any time.


USAGE
  gh preview <command> [flags]

AVAILABLE COMMANDS
  prompter:	 Execute a test program to preview the prompter

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh preview prompter`

Execute a test program to preview the prompter.
Without an argument, all prompts will be run.

Available prompt types:
- select
- multi-select
- input
- password
- confirm
- auth-token
- confirm-deletion
- input-hostname
- markdown-editor


USAGE
  gh preview prompter [prompt type] [flags]

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project`

Work with GitHub Projects.

The minimum required scope for the token is: `project`.
You can verify your token scope by running `gh auth status` and
add the `project` scope by running `gh auth refresh -s project`.


USAGE
  gh project <command> [flags]

AVAILABLE COMMANDS
  close:	 Close a project
  copy:		 Copy a project
  create:	 Create a project
  delete:	 Delete a project
  edit:		 Edit a project
  field-create:	 Create a field in a project
  field-delete:	 Delete a field in a project
  field-list:	 List the fields in a project
  item-add:	 Add a pull request or an issue to a project
  item-archive:	 Archive an item in a project
  item-create:	 Create a draft issue item in a project
  item-delete:	 Delete an item from a project by ID
  item-edit:	 Edit an item in a project
  item-list:	 List the items in a project
  link:		 Link a project to a repository or a team
  list:		 List the projects for an owner
  mark-template: Mark a project as a template
  unlink:	 Unlink a project from a repository or a team
  view:		 View a project

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh project create --owner monalisa --title "Roadmap"
  $ gh project view 1 --owner cli --web
  $ gh project field-list 1 --owner cli
  $ gh project item-list 1 --owner cli

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project close`

Close a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project close [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
      --undo		  Reopen a closed project

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Close project "1" owned by monalisa
  $ gh project close 1 --owner monalisa

  # Reopen closed project "1" owned by github
  $ gh project close 1 --owner github --undo

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project copy`

Copy a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project copy [<number>] [flags]

FLAGS
      --drafts		      Include draft issues when copying
      --format string	      Output format: {json}
  -q, --jq expression	      Filter JSON output using a jq expression
      --source-owner string   Login of the source owner. Use "@me" for the current user.
      --target-owner string   Login of the target owner. Use "@me" for the current user.
  -t, --template string	      Format JSON output using a Go template; see "gh help formatting"
      --title string	      Title for the new project

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Copy project "1" owned by monalisa to github
  $ gh project copy 1 --source-owner monalisa --target-owner github --title "a new project"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project create`

Create a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project create [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
      --title string	  Title for the project

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Create a new project owned by login monalisa
  $ gh project create --owner monalisa --title "a new project"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project delete`

Delete a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project delete [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Delete the current user's project "1"
  $ gh project delete 1 --owner "@me"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project edit`

Edit a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project edit [<number>] [flags]

FLAGS
  -d, --description string   New description of the project
      --format string	     Output format: {json}
  -q, --jq expression	     Filter JSON output using a jq expression
      --owner string	     Login of the owner. Use "@me" for the current user.
      --readme string	     New readme for the project
  -t, --template string	     Format JSON output using a Go template; see "gh help formatting"
      --title string	     New title for the project
      --visibility string    Change project visibility: {PUBLIC|PRIVATE}

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Edit the title of monalisa's project "1"
  $ gh project edit 1 --owner monalisa --title "New title"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project field-create`

Create a field in a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project field-create [<number>] [flags]

FLAGS
      --data-type string		DataType of the new field.: {TEXT|SINGLE_SELECT|DATE|NUMBER}
      --format string			Output format: {json}
  -q, --jq expression			Filter JSON output using a jq expression
      --name string			Name of the new field
      --owner string			Login of the owner. Use "@me" for the current user.
      --single-select-options strings	Options for SINGLE_SELECT data type
  -t, --template string			Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Create a field in the current user's project "1"
  $ gh project field-create 1 --owner "@me" --name "new field" --data-type "text"

  # Create a field with three options to select from for owner monalisa
  $ gh project field-create 1 --owner monalisa --name "new field" --data-type "SINGLE_SELECT" --single-select-options "one,two,three"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project field-delete`

Delete a field in a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project field-delete [flags]

FLAGS
      --format string	  Output format: {json}
      --id string	  ID of the field to delete
  -q, --jq expression	  Filter JSON output using a jq expression
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project field-list`

List the fields in a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project field-list [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
  -L, --limit int	  Maximum number of fields to fetch (default 30)
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # List fields in the current user's project "1"
  $ gh project field-list 1 --owner "@me"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project item-add`

Add a pull request or an issue to a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project item-add [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
      --url string	  URL of the issue or pull request to add to the project

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Add an item to monalisa's project "1"
  $ gh project item-add 1 --owner monalisa --url https://github.com/monalisa/myproject/issues/23

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project item-archive`

Archive an item in a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project item-archive [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
      --id string	  ID of the item to archive
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
      --undo		  Unarchive an item

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Archive an item in the current user's project "1"
  $ gh project item-archive 1 --owner "@me" --id <item-ID>

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project item-create`

Create a draft issue item in a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project item-create [<number>] [flags]

FLAGS
      --body string	  Body for the draft issue
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
      --title string	  Title for the draft issue

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Create a draft issue in the current user's project "1"
  $ gh project item-create 1 --owner "@me" --title "new item" --body "new item body"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project item-delete`

Delete an item from a project by ID

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project item-delete [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
      --id string	  ID of the item to delete
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Delete an item in the current user's project "1"
  $ gh project item-delete 1 --owner "@me" --id <item-id>

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project item-edit`

Edit either a draft issue or a project item. Both usages require the ID of the item to edit.

For non-draft issues, the ID of the project is also required, and only a single field value can be updated per invocation.

Remove project item field value using `--clear` flag.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project item-edit [flags]

FLAGS
      --body string			 Body of the draft issue item
      --clear				 Remove field value
      --date string			 Date value for the field (YYYY-MM-DD)
      --field-id string			 ID of the field to update
      --format string			 Output format: {json}
      --id string			 ID of the item to edit
      --iteration-id string		 ID of the iteration value to set on the field
  -q, --jq expression			 Filter JSON output using a jq expression
      --number float			 Number value for the field
      --project-id string		 ID of the project to which the field belongs to
      --single-select-option-id string	 ID of the single select option value to set on the field
  -t, --template string			 Format JSON output using a Go template; see "gh help formatting"
      --text string			 Text value for the field
      --title string			 Title of the draft issue item

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Edit an item's text field value
  $ gh project item-edit --id <item-id> --field-id <field-id> --project-id <project-id> --text "new text"

  # Clear an item's field value
  $ gh project item-edit --id <item-id> --field-id <field-id> --project-id <project-id> --clear

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project item-list`

List the items in a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project item-list [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
  -L, --limit int	  Maximum number of items to fetch (default 30)
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # List the items in the current users's project "1"
  $ gh project item-list 1 --owner "@me"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project link`

Link a project to a repository or a team

USAGE
  gh project link [<number>] [flags]

FLAGS
      --owner string   Login of the owner. Use "@me" for the current user.
  -R, --repo string    The repository to be linked to this project
  -T, --team string    The team to be linked to this project

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Link monalisa's project 1 to her repository "my_repo"
  $ gh project link 1 --owner monalisa --repo my_repo

  # Link monalisa's organization's project 1 to her team "my_team"
  $ gh project link 1 --owner my_organization --team my_team

  # Link monalisa's project 1 to the repository of current directory if neither --repo nor --team is specified
  $ gh project link 1

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project list`

List the projects for an owner

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project list [flags]

ALIASES
  gh project ls

FLAGS
      --closed		  Include closed projects
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
  -L, --limit int	  Maximum number of projects to fetch (default 30)
      --owner string	  Login of the owner
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  Open projects list in the browser

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # List the current user's projects
  $ gh project list

  # List the projects for org github including closed projects
  $ gh project list --owner github --closed

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project mark-template`

Mark a project as a template

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project mark-template [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the org owner.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
      --undo		  Unmark the project as a template.

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Mark the github org's project "1" as a template
  $ gh project mark-template 1 --owner "github"

  # Unmark the github org's project "1" as a template
  $ gh project mark-template 1 --owner "github" --undo

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project unlink`

Unlink a project from a repository or a team

USAGE
  gh project unlink [<number>] [flags]

FLAGS
      --owner string   Login of the owner. Use "@me" for the current user.
  -R, --repo string    The repository to be unlinked from this project
  -T, --team string    The team to be unlinked from this project

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Unlink monalisa's project 1 from her repository "my_repo"
  $ gh project unlink 1 --owner monalisa --repo my_repo

  # Unlink monalisa's organization's project 1 from her team "my_team"
  $ gh project unlink 1 --owner my_organization --team my_team

  # Unlink monalisa's project 1 from the repository of current directory if neither --repo nor --team is specified
  $ gh project unlink 1

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh project view`

View a project

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh project view [<number>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
      --owner string	  Login of the owner. Use "@me" for the current user.
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  Open a project in the browser

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # View the current user's project "1"
  $ gh project view 1

  # Open user monalisa's project "1" in the browser
  $ gh project view 1 --owner monalisa --web

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release`

Manage releases

USAGE
  gh release <command> [flags]

GENERAL COMMANDS
  create:	 Create a new release
  list:		 List releases in a repository

TARGETED COMMANDS
  delete:	 Delete a release
  delete-asset:	 Delete an asset from a release
  download:	 Download release assets
  edit:		 Edit a release
  upload:	 Upload assets to a release
  verify:	 Verify the attestation for a release
  verify-asset:	 Verify that a given asset originated from a release
  view:		 View information about a release

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release create`

Create a new GitHub Release for a repository.

A list of asset files may be given to upload to the new release. To define a
display label for an asset, append text starting with `#` after the file name.

If a matching git tag does not yet exist, one will automatically get created
from the latest state of the default branch.
Use `--target` to point to a different branch or commit for the automatic tag creation.
Use `--verify-tag` to abort the release if the tag doesn't already exist.
To fetch the new tag locally after the release, do `git fetch --tags origin`.

To create a release from an annotated git tag, first create one locally with
git, push the tag to GitHub, then run this command.
Use `--notes-from-tag` to get the release notes from the annotated git tag.
If the tag is not annotated, the commit message will be used instead.

Use `--generate-notes` to automatically generate notes using GitHub Release Notes API.
When using automatically generated release notes, a release title will also be automatically
generated unless a title was explicitly passed. Additional release notes can be prepended to
automatically generated notes by using the `--notes` flag.

By default, the release is created even if there are no new commits since the last release.
This may result in the same or duplicate release which may not be desirable in some cases.
Use `--fail-on-no-commits` to fail if no new commits are available. This flag has no
effect if there are no existing releases or this is the very first release.

## Immutable Releases

When release immutability is enabled for a repository, the following protections are enforced:
- Git tags associated with a release cannot be modified or deleted.
- Release assets cannot be modified or deleted.

Immutability is enforced only after a release is published. Draft releases can be modified
or deleted, and the associated git tags can be modified or deleted as well.

When using the `create` command to attach assets to a release, separate API calls
are made to create the release as a draft, upload the assets, and then publish the release.
Immutability protections will be enforced ONLY after the release is published.


USAGE
  gh release create [<tag>] [<filename>... | <pattern>...]

ALIASES
  gh release new

FLAGS
      --discussion-category string   Start a discussion in the specified category
  -d, --draft			     Save the release as a draft instead of publishing it
      --fail-on-no-commits	     Fail if there are no commits since the last release (no impact on the first release)
      --generate-notes		     Automatically generate title and notes for the release via GitHub Release Notes API
      --latest			     Mark this release as "Latest" (default [automatic based on date and version]). --latest=false to explicitly NOT set as latest
  -n, --notes string		     Release notes
  -F, --notes-file file		     Read release notes from file (use "-" to read from standard input)
      --notes-from-tag		     Fetch notes from the tag annotation or message of commit associated with tag
      --notes-start-tag string	     Tag to use as the starting point for generating release notes
  -p, --prerelease		     Mark the release as a prerelease
      --target branch		     Target branch or full commit SHA (default [main branch])
  -t, --title string		     Release title
      --verify-tag		     Abort in case the git tag doesn't already exist in the remote repository

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Interactively create a release
  $ gh release create

  # Interactively create a release from specific tag
  $ gh release create v1.2.3

  # Non-interactively create a release
  $ gh release create v1.2.3 --notes "bugfix release"

  # Use automatically generated via GitHub Release Notes API release notes
  $ gh release create v1.2.3 --generate-notes

  # Use release notes from a file
  $ gh release create v1.2.3 -F release-notes.md

  # Use tag annotation or associated commit message as notes
  $ gh release create v1.2.3 --notes-from-tag

  # Don't mark the release as latest
  $ gh release create v1.2.3 --latest=false

  # Upload all tarballs in a directory as release assets
  $ gh release create v1.2.3 ./dist/*.tgz

  # Upload a release asset with a display label
  $ gh release create v1.2.3 '/path/to/asset.zip#My display label'

  # Create a release and start a discussion
  $ gh release create v1.2.3 --discussion-category "General"

  # Create a release only if there are new commits available since the last release
  $ gh release create v1.2.3 --fail-on-no-commits

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release delete`

Delete a release

USAGE
  gh release delete <tag> [flags]

FLAGS
      --cleanup-tag   Delete the specified tag in addition to its release
  -y, --yes	      Skip the confirmation prompt

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release delete-asset`

Delete an asset from a release

USAGE
  gh release delete-asset <tag> <asset-name> [flags]

FLAGS
  -y, --yes   Skip the confirmation prompt

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release download`

Download assets from a GitHub release.

Without an explicit tag name argument, assets are downloaded from the
latest release in the project. In this case, `--pattern` or `--archive`
is required.


USAGE
  gh release download [<tag>] [flags]

FLAGS
  -A, --archive format	      Download the source code archive in the specified format (zip or tar.gz)
      --clobber		      Overwrite existing files of the same name
  -D, --dir directory	      The directory to download files into (default ".")
  -O, --output file	      The file to write a single asset to (use "-" to write to standard output)
  -p, --pattern stringArray   Download only assets that match a glob pattern
      --skip-existing	      Skip downloading when files of the same name exist

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Download all assets from a specific release
  $ gh release download v1.2.3

  # Download only Debian packages for the latest release
  $ gh release download --pattern '*.deb'

  # Specify multiple file patterns
  $ gh release download -p '*.deb' -p '*.rpm'

  # Download the archive of the source code for a release
  $ gh release download v1.2.3 --archive=zip

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release edit`

Edit a release

USAGE
  gh release edit <tag>

FLAGS
      --discussion-category string   Start a discussion in the specified category when publishing a draft
      --draft			     Save the release as a draft instead of publishing it
      --latest			     Explicitly mark the release as "Latest"
  -n, --notes string		     Release notes
  -F, --notes-file file		     Read release notes from file (use "-" to read from standard input)
      --prerelease		     Mark the release as a prerelease
      --tag string		     The name of the tag
      --target branch		     Target branch or full commit SHA (default [main branch])
  -t, --title string		     Release title
      --verify-tag		     Abort in case the git tag doesn't already exist in the remote repository

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Publish a release that was previously a draft
  $ gh release edit v1.0 --draft=false

  # Update the release notes from the content of a file
  $ gh release edit v1.0 --notes-file /path/to/release_notes.md

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release list`

List releases in a repository

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh release list [flags]

ALIASES
  gh release ls

FLAGS
      --exclude-drafts	       Exclude draft releases
      --exclude-pre-releases   Exclude pre-releases
  -q, --jq expression	       Filter JSON output using a jq expression
      --json fields	       Output JSON with the specified fields
  -L, --limit int	       Maximum number of items to fetch (default 30)
  -O, --order string	       Order of releases returned: {asc|desc} (default "desc")
  -t, --template string	       Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  createdAt, isDraft, isImmutable, isLatest, isPrerelease, name, publishedAt,
  tagName

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release upload`

Upload asset files to a GitHub Release.

To define a display label for an asset, append text starting with `#` after the
file name.


USAGE
  gh release upload <tag> <files>... [flags]

FLAGS
  --clobber   Overwrite existing assets of the same name

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release verify`

Verify that a GitHub Release is accompanied by a valid cryptographically signed attestation.

An attestation is a claim made by GitHub regarding a release and its assets.

This command checks that the specified release (or the latest release, if no tag is given) has a valid attestation.
It fetches the attestation for the release and prints metadata about all assets referenced in the attestation, including their digests.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh release verify [<tag>] [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Verify the latest release
  gh release verify

  # Verify a specific release by tag
  gh release verify v1.2.3

  # Verify a specific release by tag and output the attestation in JSON format
  gh release verify v1.2.3 --format json

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release verify-asset`

Verify that a given asset file originated from a specific GitHub Release using cryptographically signed attestations.

An attestation is a claim made by GitHub regarding a release and its assets.

		This command checks that the asset you provide matches a valid attestation for the specified release (or the latest release, if no tag is given).
It ensures the asset's integrity by validating that the asset's digest matches the subject in the attestation and that the attestation is associated with the release.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh release verify-asset [<tag>] <file-path> [flags]

FLAGS
      --format string	  Output format: {json}
  -q, --jq expression	  Filter JSON output using a jq expression
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Verify an asset from the latest release
  $ gh release verify-asset ./dist/my-asset.zip

  # Verify an asset from a specific release tag
  $ gh release verify-asset v1.2.3 ./dist/my-asset.zip

  # Verify an asset from a specific release tag and output the attestation in JSON format
  $ gh release verify-asset v1.2.3 ./dist/my-asset.zip --format json

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh release view`

View information about a GitHub Release.

Without an explicit tag name argument, the latest release in the project
is shown.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh release view [<tag>] [flags]

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  Open the release in the browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  apiUrl, assets, author, body, createdAt, databaseId, id, isDraft, isImmutable,
  isPrerelease, name, publishedAt, tagName, tarballUrl, targetCommitish,
  uploadUrl, url, zipballUrl

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo`

Work with GitHub repositories.

USAGE
  gh repo <command> [flags]

GENERAL COMMANDS
  create:	 Create a new repository
  list:		 List repositories owned by user or organization

TARGETED COMMANDS
  archive:	 Archive a repository
  autolink:	 Manage autolink references
  clone:	 Clone a repository locally
  delete:	 Delete a repository
  deploy-key:	 Manage deploy keys in a repository
  edit:		 Edit repository settings
  fork:		 Create a fork of a repository
  gitignore:	 List and view available repository gitignore templates
  license:	 Explore repository licenses
  rename:	 Rename a repository
  set-default:	 Configure default repository for this directory
  sync:		 Sync a repository
  unarchive:	 Unarchive a repository
  view:		 View a repository

INHERITED FLAGS
  --help   Show help for command

ARGUMENTS
  A repository can be supplied as an argument in any of the following formats:
  - "OWNER/REPO"
  - by URL, e.g. "https://github.com/OWNER/REPO"

EXAMPLES
  $ gh repo create
  $ gh repo clone cli/cli
  $ gh repo view --web

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo archive`

Archive a GitHub repository.

With no argument, archives the current repository.

USAGE
  gh repo archive [<repository>] [flags]

FLAGS
  -y, --yes   Skip the confirmation prompt

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo autolink`

Autolinks link issues, pull requests, commit messages, and release descriptions to external third-party services.

Autolinks require `admin` role to view or manage.

For more information, see <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/configuring-autolinks-to-reference-external-resources>


USAGE
  gh repo autolink <command> [flags]

AVAILABLE COMMANDS
  create:	 Create a new autolink reference
  delete:	 Delete an autolink reference
  list:		 List autolink references for a GitHub repository
  view:		 View an autolink reference

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo autolink create`

Create a new autolink reference for a repository.

The `keyPrefix` argument specifies the prefix that will generate a link when it is appended by certain characters.

The `urlTemplate` argument specifies the target URL that will be generated when the keyPrefix is found, which
must contain `<num>` variable for the reference number.

By default, autolinks are alphanumeric with `--numeric` flag used to create a numeric autolink.

The `<num>` variable behavior differs depending on whether the autolink is alphanumeric or numeric:

- alphanumeric: matches `A-Z` (case insensitive), `0-9`, and `-`
- numeric: matches `0-9`

If the template contains multiple instances of `<num>`, only the first will be replaced.


USAGE
  gh repo autolink create <keyPrefix> <urlTemplate> [flags]

ALIASES
  gh repo autolink new

FLAGS
  -n, --numeric	  Mark autolink as numeric

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Create an alphanumeric autolink to example.com for the key prefix "TICKET-".
  # Generates https://example.com/TICKET?query=123abc from "TICKET-123abc".
  $ gh repo autolink create TICKET- "https://example.com/TICKET?query=<num>"

  # Create a numeric autolink to example.com for the key prefix "STORY-".
  # Generates https://example.com/STORY?id=123 from "STORY-123".
  $ gh repo autolink create STORY- "https://example.com/STORY?id=<num>" --numeric

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo autolink delete`

Delete an autolink reference for a repository.

USAGE
  gh repo autolink delete <id> [flags]

FLAGS
  --yes	  Confirm deletion without prompting

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo autolink list`

Gets all autolink references that are configured for a repository.

Information about autolinks is only available to repository administrators.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh repo autolink list [flags]

ALIASES
  gh repo autolink ls

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  List autolink references in the web browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  id, isAlphanumeric, keyPrefix, urlTemplate

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo autolink view`

View an autolink reference for a repository.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh repo autolink view <id> [flags]

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  id, isAlphanumeric, keyPrefix, urlTemplate

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo clone`

Clone a GitHub repository locally. Pass additional `git clone` flags by listing
them after `--`.

If the `OWNER/` portion of the `OWNER/REPO` repository argument is omitted, it
defaults to the name of the authenticating user.

When a protocol scheme is not provided in the repository argument, the `git_protocol` will be
chosen from your configuration, which can be checked via `gh config get git_protocol`. If the protocol
scheme is provided, the repository will be cloned using the specified protocol.

If the repository is a fork, its parent repository will be added as an additional
git remote called `upstream`. The remote name can be configured using `--upstream-remote-name`.
The `--upstream-remote-name` option supports an `@owner` value which will name
the remote after the owner of the parent repository.

If the repository is a fork, its parent repository will be set as the default remote repository.


USAGE
  gh repo clone <repository> [<directory>] [-- <gitflags>...]

FLAGS
  -u, --upstream-remote-name string   Upstream remote name when cloning a fork (default "upstream")

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Clone a repository from a specific org
  $ gh repo clone cli/cli

  # Clone a repository from your own account
  $ gh repo clone myrepo

  # Clone a repo, overriding git protocol configuration
  $ gh repo clone https://github.com/cli/cli
  $ gh repo clone git@github.com:cli/cli.git

  # Clone a repository to a custom directory
  $ gh repo clone cli/cli workspace/cli

  # Clone a repository with additional git clone flags
  $ gh repo clone cli/cli -- --depth=1

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo create`

Create a new GitHub repository.

To create a repository interactively, use `gh repo create` with no arguments.

To create a remote repository non-interactively, supply the repository name and one of `--public`, `--private`, or `--internal`.
Pass `--clone` to clone the new repository locally.

If the `OWNER/` portion of the `OWNER/REPO` name argument is omitted, it
defaults to the name of the authenticating user.

To create a remote repository from an existing local repository, specify the source directory with `--source`.
By default, the remote repository name will be the name of the source directory.

Pass `--push` to push any local commits to the new repository. If the repo is bare, this will mirror all refs.

For language or platform .gitignore templates to use with `--gitignore`, <https://github.com/github/gitignore>.

For license keywords to use with `--license`, run `gh repo license list` or visit <https://choosealicense.com>.

The repo is created with the configured repository default branch, see <https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-personal-account-on-github/managing-user-account-settings/managing-the-default-branch-name-for-your-repositories>.


USAGE
  gh repo create [<name>] [flags]

ALIASES
  gh repo new

FLAGS
      --add-readme	       Add a README file to the new repository
  -c, --clone		       Clone the new repository to the current directory
  -d, --description string     Description of the repository
      --disable-issues	       Disable issues in the new repository
      --disable-wiki	       Disable wiki in the new repository
  -g, --gitignore string       Specify a gitignore template for the repository
  -h, --homepage URL	       Repository home page URL
      --include-all-branches   Include all branches from template repository
      --internal	       Make the new repository internal
  -l, --license string	       Specify an Open Source License for the repository
      --private		       Make the new repository private
      --public		       Make the new repository public
      --push		       Push local commits to the new repository
  -r, --remote string	       Specify remote name for the new repository
  -s, --source string	       Specify path to local repository to use as source
  -t, --team name	       The name of the organization team to be granted access
  -p, --template repository    Make the new repository based on a template repository

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Create a repository interactively
  $ gh repo create

  # Create a new remote repository and clone it locally
  $ gh repo create my-project --public --clone

  # Create a new remote repository in a different organization
  $ gh repo create my-org/my-project --public

  # Create a remote repository from the current directory
  $ gh repo create my-project --private --source=. --remote=upstream

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo delete`

Delete a GitHub repository.

With no argument, deletes the current repository. Otherwise, deletes the specified repository.

For safety, when no repository argument is provided, the `--yes` flag is ignored
and you will be prompted for confirmation. To delete the current repository non-interactively,
specify it explicitly (e.g., `gh repo delete owner/repo --yes`).

Deletion requires authorization with the `delete_repo` scope.
To authorize, run `gh auth refresh -s delete_repo`


USAGE
  gh repo delete [<repository>] [flags]

FLAGS
  --yes	  Confirm deletion without prompting

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo deploy-key`

Manage deploy keys in a repository

USAGE
  gh repo deploy-key <command> [flags]

AVAILABLE COMMANDS
  add:		 Add a deploy key to a GitHub repository
  delete:	 Delete a deploy key from a GitHub repository
  list:		 List deploy keys in a GitHub repository

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo deploy-key add`

Add a deploy key to a GitHub repository.

Note that any key added by gh will be associated with the current authentication token.
If you de-authorize the GitHub CLI app or authentication token from your account, any
deploy keys added by GitHub CLI will be removed as well.


USAGE
  gh repo deploy-key add <key-file> [flags]

FLAGS
  -w, --allow-write    Allow write access for the key
  -t, --title string   Title of the new key

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Generate a passwordless SSH key and add it as a deploy key to a repository
  $ ssh-keygen -t ed25519 -C "my description" -N "" -f ~/.ssh/gh-test
  $ gh repo deploy-key add ~/.ssh/gh-test.pub

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo deploy-key delete`

Delete a deploy key from a GitHub repository

USAGE
  gh repo deploy-key delete <key-id> [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo deploy-key list`

List deploy keys in a GitHub repository

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh repo deploy-key list [flags]

ALIASES
  gh repo deploy-key ls

FLAGS
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  createdAt, id, key, readOnly, title

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo edit`

Edit repository settings.

To toggle a setting off, use the `--<flag>=false` syntax.

Changing repository visibility can have unexpected consequences including but not limited to:

- Losing stars and watchers, affecting repository ranking
- Detaching public forks from the network
- Disabling push rulesets
- Allowing access to GitHub Actions history and logs

When the `--visibility` flag is used, `--accept-visibility-change-consequences` flag is required.

For information on all the potential consequences, see <https://gh.io/setting-repository-visibility>.


USAGE
  gh repo edit [<repository>] [flags]

FLAGS
      --accept-visibility-change-consequences	 Accept the consequences of changing the repository visibility
      --add-topic strings			 Add repository topic
      --allow-forking				 Allow forking of an organization repository
      --allow-update-branch			 Allow a pull request head branch that is behind its base branch to be updated
      --default-branch name			 Set the default branch name for the repository
      --delete-branch-on-merge			 Delete head branch when pull requests are merged
  -d, --description string			 Description of the repository
      --enable-advanced-security		 Enable advanced security in the repository
      --enable-auto-merge			 Enable auto-merge functionality
      --enable-discussions			 Enable discussions in the repository
      --enable-issues				 Enable issues in the repository
      --enable-merge-commit			 Enable merging pull requests via merge commit
      --enable-projects				 Enable projects in the repository
      --enable-rebase-merge			 Enable merging pull requests via rebase
      --enable-secret-scanning			 Enable secret scanning in the repository
      --enable-secret-scanning-push-protection	 Enable secret scanning push protection in the repository. Secret scanning must be enabled first
      --enable-squash-merge			 Enable merging pull requests via squashed commit
      --enable-wiki				 Enable wiki in the repository
  -h, --homepage URL				 Repository home page URL
      --remove-topic strings			 Remove repository topic
      --template				 Make the repository available as a template repository
      --visibility string			 Change the visibility of the repository to {public,private,internal}

INHERITED FLAGS
  --help   Show help for command

ARGUMENTS
  A repository can be supplied as an argument in any of the following formats:
  - "OWNER/REPO"
  - by URL, e.g. "https://github.com/OWNER/REPO"

EXAMPLES
  # Enable issues and wiki
  $ gh repo edit --enable-issues --enable-wiki

  # Disable projects
  $ gh repo edit --enable-projects=false

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo fork`

Create a fork of a repository.

With no argument, creates a fork of the current repository. Otherwise, forks
the specified repository.

By default, the new fork is set to be your `origin` remote and any existing
origin remote is renamed to `upstream`. To alter this behavior, you can set
a name for the new fork's remote with `--remote-name`.

The `upstream` remote will be set as the default remote repository.

Additional `git clone` flags can be passed after `--`.


USAGE
  gh repo fork [<repository>] [-- <gitflags>...] [flags]

FLAGS
  --clone		  Clone the fork
  --default-branch-only	  Only include the default branch in the fork
  --fork-name string	  Rename the forked repository
  --org string		  Create the fork in an organization
  --remote		  Add a git remote for the fork
  --remote-name string	  Specify the name for the new remote (default "origin")

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo gitignore`

List and view available repository gitignore templates

USAGE
  gh repo gitignore <command> [flags]

AVAILABLE COMMANDS
  list:		 List available repository gitignore templates
  view:		 View an available repository gitignore template

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo gitignore list`

List available repository gitignore templates

USAGE
  gh repo gitignore list [flags]

ALIASES
  gh repo gitignore ls

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo gitignore view`

View an available repository `.gitignore` template.

`<template>` is a case-sensitive `.gitignore` template name.

For a list of available templates, run `gh repo gitignore list`.


USAGE
  gh repo gitignore view <template> [flags]

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # View the Go gitignore template
  $ gh repo gitignore view Go

  # View the Python gitignore template
  $ gh repo gitignore view Python

  # Create a new .gitignore file using the Go template
  $ gh repo gitignore view Go > .gitignore

  # Create a new .gitignore file using the Python template
  $ gh repo gitignore view Python > .gitignore

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo license`

Explore repository licenses

USAGE
  gh repo license <command> [flags]

AVAILABLE COMMANDS
  list:		 List common repository licenses
  view:		 View a specific repository license

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo license list`

List common repository licenses.

For even more licenses, visit <https://choosealicense.com/appendix>


USAGE
  gh repo license list [flags]

ALIASES
  gh repo license ls

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo license view`

View a specific repository license by license key or SPDX ID.

Run `gh repo license list` to see available commonly used licenses. For even more licenses, visit <https://choosealicense.com/appendix>.


USAGE
  gh repo license view {<license-key> | <spdx-id>} [flags]

FLAGS
  -w, --web   Open https://choosealicense.com/ in the browser

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # View the MIT license from SPDX ID
  $ gh repo license view MIT

  # View the MIT license from license key
  $ gh repo license view mit

  # View the GNU AGPL-3.0 license from SPDX ID
  $ gh repo license view AGPL-3.0

  # View the GNU AGPL-3.0 license from license key
  $ gh repo license view agpl-3.0

  # Create a LICENSE.md with the MIT license
  $ gh repo license view MIT > LICENSE.md

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo list`

List repositories owned by a user or organization.

Note that the list will only include repositories owned by the provided argument,
and the `--fork` or `--source` flags will not traverse ownership boundaries. For example,
when listing the forks in an organization, the output would not include those owned by individual users.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh repo list [<owner>] [flags]

ALIASES
  gh repo ls

FLAGS
      --archived	    Show only archived repositories
      --fork		    Show only forks
  -q, --jq expression	    Filter JSON output using a jq expression
      --json fields	    Output JSON with the specified fields
  -l, --language string	    Filter by primary coding language
  -L, --limit int	    Maximum number of repositories to list (default 30)
      --no-archived	    Omit archived repositories
      --source		    Show only non-forks
  -t, --template string	    Format JSON output using a Go template; see "gh help formatting"
      --topic strings	    Filter by topic
      --visibility string   Filter by repository visibility: {public|private|internal}

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  archivedAt, assignableUsers, codeOfConduct, contactLinks, createdAt,
  defaultBranchRef, deleteBranchOnMerge, description, diskUsage, forkCount,
  fundingLinks, hasDiscussionsEnabled, hasIssuesEnabled, hasProjectsEnabled,
  hasWikiEnabled, homepageUrl, id, isArchived, isBlankIssuesEnabled, isEmpty,
  isFork, isInOrganization, isMirror, isPrivate, isSecurityPolicyEnabled,
  isTemplate, isUserConfigurationRepository, issueTemplates, issues, labels,
  languages, latestRelease, licenseInfo, mentionableUsers, mergeCommitAllowed,
  milestones, mirrorUrl, name, nameWithOwner, openGraphImageUrl, owner, parent,
  primaryLanguage, projects, projectsV2, pullRequestTemplates, pullRequests,
  pushedAt, rebaseMergeAllowed, repositoryTopics, securityPolicyUrl,
  squashMergeAllowed, sshUrl, stargazerCount, templateRepository, updatedAt, url,
  usesCustomOpenGraphImage, viewerCanAdminister, viewerDefaultCommitEmail,
  viewerDefaultMergeMethod, viewerHasStarred, viewerPermission,
  viewerPossibleCommitEmails, viewerSubscription, visibility, watchers

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo rename`

Rename a GitHub repository.

`<new-name>` is the desired repository name without the owner.

By default, the current repository is renamed. Otherwise, the repository specified
with `--repo` is renamed.

To transfer repository ownership to another user account or organization,
you must follow additional steps on `github.com`.

For more information on transferring repository ownership, see:
<https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository>


USAGE
  gh repo rename [<new-name>] [flags]

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format
  -y, --yes			 Skip the confirmation prompt

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Rename the current repository (foo/bar -> foo/baz)
  $ gh repo rename baz

  # Rename the specified repository (qux/quux -> qux/baz)
  $ gh repo rename -R qux/quux baz

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo set-default`

This command sets the default remote repository to use when querying the
GitHub API for the locally cloned repository.

gh uses the default repository for things like:

 - viewing and creating pull requests
 - viewing and creating issues
 - viewing and creating releases
 - working with GitHub Actions

### NOTE: gh does not use the default repository for managing repository and environment secrets.

USAGE
  gh repo set-default [<repository>] [flags]

FLAGS
  -u, --unset	Unset the current default repository
  -v, --view	View the current default repository

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Interactively select a default repository
  $ gh repo set-default

  # Set a repository explicitly
  $ gh repo set-default owner/repo

  # View the current default repository
  $ gh repo set-default --view

  # Show more repository options in the interactive picker
  $ git remote add newrepo https://github.com/owner/repo
  $ gh repo set-default

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo sync`

Sync destination repository from source repository. Syncing uses the default branch
of the source repository to update the matching branch on the destination
repository so they are equal. A fast forward update will be used except when the
`--force` flag is specified, then the two branches will
be synced using a hard reset.

Without an argument, the local repository is selected as the destination repository.

The source repository is the parent of the destination repository by default.
This can be overridden with the `--source` flag.


USAGE
  gh repo sync [<destination-repository>] [flags]

FLAGS
  -b, --branch string	Branch to sync (default [default branch])
      --force		Hard reset the branch of the destination repository to match the source repository
  -s, --source string	Source repository

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  # Sync local repository from remote parent
  $ gh repo sync

  # Sync local repository from remote parent on specific branch
  $ gh repo sync --branch v1

  # Sync remote fork from its parent
  $ gh repo sync owner/cli-fork

  # Sync remote repository from another remote repository
  $ gh repo sync owner/repo --source owner2/repo2

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo unarchive`

Unarchive a GitHub repository.

With no argument, unarchives the current repository.

USAGE
  gh repo unarchive [<repository>] [flags]

FLAGS
  -y, --yes   Skip the confirmation prompt

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh repo view`

Display the description and the README of a GitHub repository.

With no argument, the repository for the current directory is displayed.

With `--web`, open the repository in a web browser instead.

With `--branch`, view a specific branch of the repository.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh repo view [<repository>] [flags]

FLAGS
  -b, --branch string	  View a specific branch of the repository
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -w, --web		  Open a repository in the browser

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  archivedAt, assignableUsers, codeOfConduct, contactLinks, createdAt,
  defaultBranchRef, deleteBranchOnMerge, description, diskUsage, forkCount,
  fundingLinks, hasDiscussionsEnabled, hasIssuesEnabled, hasProjectsEnabled,
  hasWikiEnabled, homepageUrl, id, isArchived, isBlankIssuesEnabled, isEmpty,
  isFork, isInOrganization, isMirror, isPrivate, isSecurityPolicyEnabled,
  isTemplate, isUserConfigurationRepository, issueTemplates, issues, labels,
  languages, latestRelease, licenseInfo, mentionableUsers, mergeCommitAllowed,
  milestones, mirrorUrl, name, nameWithOwner, openGraphImageUrl, owner, parent,
  primaryLanguage, projects, projectsV2, pullRequestTemplates, pullRequests,
  pushedAt, rebaseMergeAllowed, repositoryTopics, securityPolicyUrl,
  squashMergeAllowed, sshUrl, stargazerCount, templateRepository, updatedAt, url,
  usesCustomOpenGraphImage, viewerCanAdminister, viewerDefaultCommitEmail,
  viewerDefaultMergeMethod, viewerHasStarred, viewerPermission,
  viewerPossibleCommitEmails, viewerSubscription, visibility, watchers

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ruleset`

Repository rulesets are a way to define a set of rules that apply to a repository.
These commands allow you to view information about them.


USAGE
  gh ruleset <command> [flags]

ALIASES
  gh rs

AVAILABLE COMMANDS
  check:	 View rules that would apply to a given branch
  list:		 List rulesets for a repository or organization
  view:		 View information about a ruleset

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh ruleset list
  $ gh ruleset view --repo OWNER/REPO --web
  $ gh ruleset check branch-name

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ruleset check`

View information about GitHub rules that apply to a given branch.

The provided branch name does not need to exist; rules will be displayed that would apply
to a branch with that name. All rules are returned regardless of where they are configured.

If no branch name is provided, then the current branch will be used.

The `--default` flag can be used to view rules that apply to the default branch of the
repository.


USAGE
  gh ruleset check [<branch>] [flags]

FLAGS
      --default	  Check rules on default branch
  -w, --web	  Open the branch rules page in a web browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # View all rules that apply to the current branch
  $ gh ruleset check

  # View all rules that apply to a branch named "my-branch" in a different repository
  $ gh ruleset check my-branch --repo owner/repo

  # View all rules that apply to the default branch in a different repository
  $ gh ruleset check --default --repo owner/repo

  # View a ruleset configured in a different repository or any of its parents
  $ gh ruleset view 23 --repo owner/repo

  # View an organization-level ruleset
  $ gh ruleset view 23 --org my-org

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ruleset list`

List GitHub rulesets for a repository or organization.

If no options are provided, the current repository's rulesets are listed. You can query a different
repository's rulesets by using the `--repo` flag. You can also use the `--org` flag to list rulesets
configured for the provided organization.

Use the `--parents` flag to control whether rulesets configured at higher levels that also apply to the provided
repository or organization should be returned. The default is `true`.

Your access token must have the `admin:org` scope to use the `--org` flag, which can be granted by running `gh auth refresh -s admin:org`.


USAGE
  gh ruleset list [flags]

ALIASES
  gh rs ls, gh ruleset ls

FLAGS
  -L, --limit int    Maximum number of rulesets to list (default 30)
  -o, --org string   List organization-wide rulesets for the provided organization
  -p, --parents	     Whether to include rulesets configured at higher levels that also apply (default true)
  -w, --web	     Open the list of rulesets in the web browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # List rulesets in the current repository
  $ gh ruleset list

  # List rulesets in a different repository, including those configured at higher levels
  $ gh ruleset list --repo owner/repo --parents

  # List rulesets in an organization
  $ gh ruleset list --org org-name

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ruleset view`

View information about a GitHub ruleset.

If no ID is provided, an interactive prompt will be used to choose
the ruleset to view.

Use the `--parents` flag to control whether rulesets configured at higher
levels that also apply to the provided repository or organization should
be returned. The default is `true`.


USAGE
  gh ruleset view [<ruleset-id>] [flags]

FLAGS
  -o, --org string   Organization name if the provided ID is an organization-level ruleset
  -p, --parents	     Whether to include rulesets configured at higher levels that also apply (default true)
  -w, --web	     Open the ruleset in the browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Interactively choose a ruleset to view from all rulesets that apply to the current repository
  $ gh ruleset view

  # Interactively choose a ruleset to view from only rulesets configured in the current repository
  $ gh ruleset view --no-parents

  # View a ruleset configured in the current repository or any of its parents
  $ gh ruleset view 43

  # View a ruleset configured in a different repository or any of its parents
  $ gh ruleset view 23 --repo owner/repo

  # View an organization-level ruleset
  $ gh ruleset view 23 --org my-org

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run`

List, view, and watch recent workflow runs from GitHub Actions.

USAGE
  gh run <command> [flags]

AVAILABLE COMMANDS
  cancel:	 Cancel a workflow run
  delete:	 Delete a workflow run
  download:	 Download artifacts generated by a workflow run
  list:		 List recent workflow runs
  rerun:	 Rerun a run
  view:		 View a summary of a workflow run
  watch:	 Watch a run until it completes, showing its progress

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run cancel`

Cancel a workflow run

USAGE
  gh run cancel [<run-id>] [flags]

FLAGS
  --force   Force cancel a workflow run

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run delete`

Delete a workflow run

USAGE
  gh run delete [<run-id>] [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Interactively select a run to delete
  $ gh run delete

  # Delete a specific run
  $ gh run delete 12345

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run download`

Download artifacts generated by a GitHub Actions workflow run.

The contents of each artifact will be extracted under separate directories based on
the artifact name. If only a single artifact is specified, it will be extracted into
the current directory.

By default, this command downloads the latest artifact created and uploaded through
GitHub Actions. Because workflows can delete or overwrite artifacts, `<run-id>`
must be used to select an artifact from a specific workflow run.


USAGE
  gh run download [<run-id>] [flags]

FLAGS
  -D, --dir string	      The directory to download artifacts into (default ".")
  -n, --name stringArray      Download artifacts that match any of the given names
  -p, --pattern stringArray   Download artifacts that match a glob pattern

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Download all artifacts generated by a workflow run
  $ gh run download <run-id>

  # Download a specific artifact within a run
  $ gh run download <run-id> -n <name>

  # Download specific artifacts across all runs in a repository
  $ gh run download -n <name1> -n <name2>

  # Select artifacts to download interactively
  $ gh run download

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run list`

List recent workflow runs.

Note that providing the `workflow_name` to the `-w` flag will not fetch disabled workflows.
Also pass the `-a` flag to fetch disabled workflow runs using the `workflow_name` and the `-w` flag.

Runs created by organization and enterprise ruleset workflows will not display a workflow name due to GitHub API limitations.

To see runs associated with a pull request, users should run `gh pr checks`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh run list [flags]

ALIASES
  gh run ls

FLAGS
  -a, --all		  Include disabled workflows
  -b, --branch string	  Filter runs by branch
  -c, --commit SHA	  Filter runs by the SHA of the commit
      --created date	  Filter runs by the date it was created
  -e, --event event	  Filter runs by which event triggered the run
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -L, --limit int	  Maximum number of runs to fetch (default 20)
  -s, --status string	  Filter runs by status: {queued|completed|in_progress|requested|waiting|pending|action_required|cancelled|failure|neutral|skipped|stale|startup_failure|success|timed_out}
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -u, --user string	  Filter runs by user who triggered the run
  -w, --workflow string	  Filter runs by workflow

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  attempt, conclusion, createdAt, databaseId, displayTitle, event, headBranch,
  headSha, name, number, startedAt, status, updatedAt, url, workflowDatabaseId,
  workflowName

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run rerun`

Rerun an entire run, only failed jobs, or a specific job from a run.

Note that due to historical reasons, the `--job` flag may not take what you expect.
Specifically, when navigating to a job in the browser, the URL looks like this:
`https://github.com/<owner>/<repo>/actions/runs/<run-id>/jobs/<number>`.

However, this `<number>` should not be used with the `--job` flag and will result in the
API returning `404 NOT FOUND`. Instead, you can get the correct job IDs using the following command:

	gh run view <run-id> --json jobs --jq '.jobs[] | {name, databaseId}'

You will need to use databaseId field for triggering job re-runs.


USAGE
  gh run rerun [<run-id>] [flags]

FLAGS
  -d, --debug	     Rerun with debug logging
      --failed	     Rerun only failed jobs, including dependencies
  -j, --job string   Rerun a specific job ID from a run, including dependencies

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run view`

View a summary of a workflow run.

Due to platform limitations, `gh` may not always be able to associate jobs with their
corresponding logs when using the primary method of fetching logs in zip format.

In such cases, `gh` will attempt to fetch logs for each job individually via the API.
This fallback is slower and more resource-intensive. If more than 25 job logs are missing,
the operation will fail with an error.

Additionally, due to similar platform constraints, some log lines may not be
associated with a specific step within a job. In these cases, the step name will
appear as `UNKNOWN STEP` in the log output.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh run view [<run-id>] [flags]

FLAGS
  -a, --attempt uint	  The attempt number of the workflow run
      --exit-status	  Exit with non-zero status if run failed
  -j, --job string	  View a specific job ID from a run
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
      --log		  View full log for either a run or specific job
      --log-failed	  View the log for any failed steps in a run or specific job
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -v, --verbose		  Show job steps
  -w, --web		  Open run in the browser

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  attempt, conclusion, createdAt, databaseId, displayTitle, event, headBranch,
  headSha, jobs, name, number, startedAt, status, updatedAt, url,
  workflowDatabaseId, workflowName

EXAMPLES
  # Interactively select a run to view, optionally selecting a single job
  $ gh run view

  # View a specific run
  $ gh run view 12345

  # View a specific run with specific attempt number
  $ gh run view 12345 --attempt 3

  # View a specific job within a run
  $ gh run view --job 456789

  # View the full log for a specific job
  $ gh run view --log --job 456789

  # Exit non-zero if a run failed
  $ gh run view 0451 --exit-status && echo "run pending or passed"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh run watch`

Watch a run until it completes, showing its progress.

By default, all steps are displayed. The `--compact` option can be used to only
show the relevant/failed steps.

This command does not support authenticating via fine grained PATs
as it is not currently possible to create a PAT with the `checks:read` permission.


USAGE
  gh run watch <run-id> [flags]

FLAGS
      --compact	       Show only relevant/failed steps
      --exit-status    Exit with non-zero status if run fails
  -i, --interval int   Refresh interval in seconds (default 3)

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Watch a run until it's done
  $ gh run watch

  # Watch a run in compact mode
  $ gh run watch --compact

  # Run some other command when the run is finished
  $ gh run watch && notify-send 'run is done!'

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh search`

Search across all of GitHub.

Excluding search results that match a qualifier

In a browser, the GitHub search syntax supports excluding results that match a search qualifier
by prefixing the qualifier with a hyphen. For example, to search for issues that
do not have the label "bug", you would use `-label:bug` as a search qualifier.

`gh` supports this syntax in `gh search` as well, but it requires extra
command line arguments to avoid the hyphen being interpreted as a command line flag because it begins with a hyphen.

On Unix-like systems, you can use the `--` argument to indicate that
the arguments that follow are not a flag, but rather a query string. For example:

$ gh search issues -- "my-search-query -label:bug"

On PowerShell, you must use both the `--%` argument and the `--` argument to
produce the same effect. For example:

$ gh --% search issues -- "my search query -label:bug"

See the following for more information:
- GitHub search syntax: <https://docs.github.com/en/search-github/getting-started-with-searching-on-github/understanding-the-search-syntax#exclude-results-that-match-a-qualifier>
- The PowerShell stop parse flag `--%`: <https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing?view=powershell-7.5#the-stop-parsing-token>
- The Unix-like `--` argument: <https://www.gnu.org/software/bash/manual/bash.html#Shell-Builtin-Commands-1>


USAGE
  gh search <command> [flags]

AVAILABLE COMMANDS
  code:		 Search within code
  commits:	 Search for commits
  issues:	 Search for issues
  prs:		 Search for pull requests
  repos:	 Search for repositories

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh search code`

Search within code in GitHub repositories.

The search syntax is documented at:
<https://docs.github.com/search-github/searching-on-github/searching-code>

Note that these search results are powered by what is now a legacy GitHub code search engine.
The results might not match what is seen on `github.com`, and new features like regex search
are not yet available via the GitHub API.

For more information on handling search queries containing a hyphen, run `gh search --help`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh search code <query> [flags]

FLAGS
      --extension string   Filter on file extension
      --filename string	   Filter on filename
  -q, --jq expression	   Filter JSON output using a jq expression
      --json fields	   Output JSON with the specified fields
      --language string	   Filter results by language
  -L, --limit int	   Maximum number of code results to fetch (default 30)
      --match strings	   Restrict search to file contents or file path: {file|path}
      --owner strings	   Filter on owner
  -R, --repo strings	   Filter on repository
      --size string	   Filter on size range, in kilobytes
  -t, --template string	   Format JSON output using a Go template; see "gh help formatting"
  -w, --web		   Open the search query in the web browser

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  path, repository, sha, textMatches, url

EXAMPLES
  # Search code matching "react" and "lifecycle"
  $ gh search code react lifecycle

  # Search code matching "error handling"
  $ gh search code "error handling"

  # Search code matching "deque" in Python files
  $ gh search code deque --language=python

  # Search code matching "cli" in repositories owned by microsoft organization
  $ gh search code cli --owner=microsoft

  # Search code matching "panic" in the GitHub CLI repository
  $ gh search code panic --repo cli/cli

  # Search code matching keyword "lint" in package.json files
  $ gh search code lint --filename package.json

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh search commits`

Search for commits on GitHub.

The command supports constructing queries using the GitHub search syntax,
using the parameter and qualifier flags, or a combination of the two.

GitHub search syntax is documented at:
<https://docs.github.com/search-github/searching-on-github/searching-commits>

For more information on handling search queries containing a hyphen, run `gh search --help`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh search commits [<query>] [flags]

FLAGS
      --author string		 Filter by author
      --author-date date	 Filter based on authored date
      --author-email string	 Filter on author email
      --author-name string	 Filter on author name
      --committer string	 Filter by committer
      --committer-date date	 Filter based on committed date
      --committer-email string	 Filter on committer email
      --committer-name string	 Filter on committer name
      --hash string		 Filter by commit hash
  -q, --jq expression		 Filter JSON output using a jq expression
      --json fields		 Output JSON with the specified fields
  -L, --limit int		 Maximum number of commits to fetch (default 30)
      --merge			 Filter on merge commits
      --order string		 Order of commits returned, ignored unless '--sort' flag is specified: {asc|desc} (default "desc")
      --owner strings		 Filter on repository owner
      --parent string		 Filter by parent hash
  -R, --repo strings		 Filter on repository
      --sort string		 Sort fetched commits: {author-date|committer-date} (default "best-match")
  -t, --template string		 Format JSON output using a Go template; see "gh help formatting"
      --tree string		 Filter by tree hash
      --visibility strings	 Filter based on repository visibility: {public|private|internal}
  -w, --web			 Open the search query in the web browser

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  author, commit, committer, id, parents, repository, sha, url

EXAMPLES
  # Search commits matching set of keywords "readme" and "typo"
  $ gh search commits readme typo

  # Search commits matching phrase "bug fix"
  $ gh search commits "bug fix"

  # Search commits committed by user "monalisa"
  $ gh search commits --committer=monalisa

  # Search commits authored by users with name "Jane Doe"
  $ gh search commits --author-name="Jane Doe"

  # Search commits matching hash "8dd03144ffdc6c0d486d6b705f9c7fba871ee7c3"
  $ gh search commits --hash=8dd03144ffdc6c0d486d6b705f9c7fba871ee7c3

  # Search commits authored before February 1st, 2022
  $ gh search commits --author-date="<2022-02-01"

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh search issues`

Search for issues on GitHub.

The command supports constructing queries using the GitHub search syntax,
using the parameter and qualifier flags, or a combination of the two.

GitHub search syntax is documented at:
<https://docs.github.com/search-github/searching-on-github/searching-issues-and-pull-requests>

On supported GitHub hosts, advanced issue search syntax can be used in the
`--search` query. For more information about advanced issue search, see:
<https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/filtering-and-searching-issues-and-pull-requests#building-advanced-filters-for-issues>

For more information on handling search queries containing a hyphen, run `gh search --help`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh search issues [<query>] [flags]

FLAGS
      --app string	       Filter by GitHub App author
      --archived	       Filter based on the repository archived state {true|false}
      --assignee string	       Filter by assignee
      --author string	       Filter by author
      --closed date	       Filter on closed at date
      --commenter user	       Filter based on comments by user
      --comments number	       Filter on number of comments
      --created date	       Filter based on created at date
      --include-prs	       Include pull requests in results
      --interactions number    Filter on number of reactions and comments
      --involves user	       Filter based on involvement of user
  -q, --jq expression	       Filter JSON output using a jq expression
      --json fields	       Output JSON with the specified fields
      --label strings	       Filter on label
      --language string	       Filter based on the coding language
  -L, --limit int	       Maximum number of results to fetch (default 30)
      --locked		       Filter on locked conversation status
      --match strings	       Restrict search to specific field of issue: {title|body|comments}
      --mentions user	       Filter based on user mentions
      --milestone title	       Filter by milestone title
      --no-assignee	       Filter on missing assignee
      --no-label	       Filter on missing label
      --no-milestone	       Filter on missing milestone
      --no-project	       Filter on missing project
      --order string	       Order of results returned, ignored unless '--sort' flag is specified: {asc|desc} (default "desc")
      --owner strings	       Filter on repository owner
      --project owner/number   Filter on project board owner/number
      --reactions number       Filter on number of reactions
  -R, --repo strings	       Filter on repository
      --sort string	       Sort fetched results: {comments|created|interactions|reactions|reactions-+1|reactions--1|reactions-heart|reactions-smile|reactions-tada|reactions-thinking_face|updated} (default "best-match")
      --state string	       Filter based on state: {open|closed}
      --team-mentions string   Filter based on team mentions
  -t, --template string	       Format JSON output using a Go template; see "gh help formatting"
      --updated date	       Filter on last updated at date
      --visibility strings     Filter based on repository visibility: {public|private|internal}
  -w, --web		       Open the search query in the web browser

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  assignees, author, authorAssociation, body, closedAt, commentsCount, createdAt,
  id, isLocked, isPullRequest, labels, number, repository, state, title,
  updatedAt, url

EXAMPLES
  # Search issues matching set of keywords "readme" and "typo"
  $ gh search issues readme typo

  # Search issues matching phrase "broken feature"
  $ gh search issues "broken feature"

  # Search issues and pull requests in cli organization
  $ gh search issues --include-prs --owner=cli

  # Search open issues assigned to yourself
  $ gh search issues --assignee=@me --state=open

  # Search issues with numerous comments
  $ gh search issues --comments=">100"

  # Search issues without label "bug"
  $ gh search issues -- -label:bug

  # Search issues only from un-archived repositories (default is all repositories)
  $ gh search issues --owner github --archived=false

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh search prs`

Search for pull requests on GitHub.

The command supports constructing queries using the GitHub search syntax,
using the parameter and qualifier flags, or a combination of the two.

GitHub search syntax is documented at:
<https://docs.github.com/search-github/searching-on-github/searching-issues-and-pull-requests>

On supported GitHub hosts, advanced issue search syntax can be used in the
`--search` query. For more information about advanced issue search, see:
<https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/filtering-and-searching-issues-and-pull-requests#building-advanced-filters-for-issues>

For more information on handling search queries containing a hyphen, run `gh search --help`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh search prs [<query>] [flags]

FLAGS
      --app string		Filter by GitHub App author
      --archived		Filter based on the repository archived state {true|false}
      --assignee string		Filter by assignee
      --author string		Filter by author
  -B, --base string		Filter on base branch name
      --checks string		Filter based on status of the checks: {pending|success|failure}
      --closed date		Filter on closed at date
      --commenter user		Filter based on comments by user
      --comments number		Filter on number of comments
      --created date		Filter based on created at date
      --draft			Filter based on draft state
  -H, --head string		Filter on head branch name
      --interactions number	Filter on number of reactions and comments
      --involves user		Filter based on involvement of user
  -q, --jq expression		Filter JSON output using a jq expression
      --json fields		Output JSON with the specified fields
      --label strings		Filter on label
      --language string		Filter based on the coding language
  -L, --limit int		Maximum number of results to fetch (default 30)
      --locked			Filter on locked conversation status
      --match strings		Restrict search to specific field of issue: {title|body|comments}
      --mentions user		Filter based on user mentions
      --merged			Filter based on merged state
      --merged-at date		Filter on merged at date
      --milestone title		Filter by milestone title
      --no-assignee		Filter on missing assignee
      --no-label		Filter on missing label
      --no-milestone		Filter on missing milestone
      --no-project		Filter on missing project
      --order string		Order of results returned, ignored unless '--sort' flag is specified: {asc|desc} (default "desc")
      --owner strings		Filter on repository owner
      --project owner/number	Filter on project board owner/number
      --reactions number	Filter on number of reactions
  -R, --repo strings		Filter on repository
      --review string		Filter based on review status: {none|required|approved|changes_requested}
      --review-requested user	Filter on user or team requested to review
      --reviewed-by user	Filter on user who reviewed
      --sort string		Sort fetched results: {comments|reactions|reactions-+1|reactions--1|reactions-smile|reactions-thinking_face|reactions-heart|reactions-tada|interactions|created|updated} (default "best-match")
      --state string		Filter based on state: {open|closed}
      --team-mentions string	Filter based on team mentions
  -t, --template string		Format JSON output using a Go template; see "gh help formatting"
      --updated date		Filter on last updated at date
      --visibility strings	Filter based on repository visibility: {public|private|internal}
  -w, --web			Open the search query in the web browser

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  assignees, author, authorAssociation, body, closedAt, commentsCount, createdAt,
  id, isDraft, isLocked, isPullRequest, labels, number, repository, state, title,
  updatedAt, url

EXAMPLES
  # Search pull requests matching set of keywords "fix" and "bug"
  $ gh search prs fix bug

  # Search draft pull requests in cli repository
  $ gh search prs --repo=cli/cli --draft

  # Search open pull requests requesting your review
  $ gh search prs --review-requested=@me --state=open

  # Search merged pull requests assigned to yourself
  $ gh search prs --assignee=@me --merged

  # Search pull requests with numerous reactions
  $ gh search prs --reactions=">100"

  # Search pull requests without label "bug"
  $ gh search prs -- -label:bug

  # Search pull requests only from un-archived repositories (default is all repositories)
  $ gh search prs --owner github --archived=false

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh search repos`

Search for repositories on GitHub.

The command supports constructing queries using the GitHub search syntax,
using the parameter and qualifier flags, or a combination of the two.

GitHub search syntax is documented at:
<https://docs.github.com/search-github/searching-on-github/searching-for-repositories>

For more information on handling search queries containing a hyphen, run `gh search --help`.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh search repos [<query>] [flags]

FLAGS
      --archived		    Filter based on the repository archived state {true|false}
      --created date		    Filter based on created at date
      --followers number	    Filter based on number of followers
      --forks number		    Filter on number of forks
      --good-first-issues number    Filter on number of issues with the 'good first issue' label
      --help-wanted-issues number   Filter on number of issues with the 'help wanted' label
      --include-forks string	    Include forks in fetched repositories: {false|true|only}
  -q, --jq expression		    Filter JSON output using a jq expression
      --json fields		    Output JSON with the specified fields
      --language string		    Filter based on the coding language
      --license strings		    Filter based on license type
  -L, --limit int		    Maximum number of repositories to fetch (default 30)
      --match strings		    Restrict search to specific field of repository: {name|description|readme}
      --number-topics number	    Filter on number of topics
      --order string		    Order of repositories returned, ignored unless '--sort' flag is specified: {asc|desc} (default "desc")
      --owner strings		    Filter on owner
      --size string		    Filter on a size range, in kilobytes
      --sort string		    Sort fetched repositories: {forks|help-wanted-issues|stars|updated} (default "best-match")
      --stars number		    Filter on number of stars
  -t, --template string		    Format JSON output using a Go template; see "gh help formatting"
      --topic strings		    Filter on topic
      --updated date		    Filter on last updated at date
      --visibility strings	    Filter based on visibility: {public|private|internal}
  -w, --web			    Open the search query in the web browser

INHERITED FLAGS
  --help   Show help for command

JSON FIELDS
  createdAt, defaultBranch, description, forksCount, fullName, hasDownloads,
  hasIssues, hasPages, hasProjects, hasWiki, homepage, id, isArchived, isDisabled,
  isFork, isPrivate, language, license, name, openIssuesCount, owner, pushedAt,
  size, stargazersCount, updatedAt, url, visibility, watchersCount

EXAMPLES
  # Search repositories matching set of keywords "cli" and "shell"
  $ gh search repos cli shell

  # Search repositories matching phrase "vim plugin"
  $ gh search repos "vim plugin"

  # Search repositories public repos in the microsoft organization
  $ gh search repos --owner=microsoft --visibility=public

  # Search repositories with a set of topics
  $ gh search repos --topic=unix,terminal

  # Search repositories by coding language and number of good first issues
  $ gh search repos --language=go --good-first-issues=">=10"

  # Search repositories without topic "linux"
  $ gh search repos -- -topic:linux

  # Search repositories excluding archived repositories
  $ gh search repos --archived=false

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh secret`

Secrets can be set at the repository, or organization level for use in
GitHub Actions or Dependabot. User, organization, and repository secrets can be set for
use in GitHub Codespaces. Environment secrets can be set for use in
GitHub Actions. Run `gh help secret set` to learn how to get started.


USAGE
  gh secret <command> [flags]

AVAILABLE COMMANDS
  delete:	 Delete secrets
  list:		 List secrets
  set:		 Create or update secrets

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh secret delete`

Delete a secret on one of the following levels:
- repository (default): available to GitHub Actions runs or Dependabot in a repository
- environment: available to GitHub Actions runs for a deployment environment in a repository
- organization: available to GitHub Actions runs, Dependabot, or Codespaces within an organization
- user: available to Codespaces for your user


USAGE
  gh secret delete <secret-name> [flags]

ALIASES
  gh secret remove

FLAGS
  -a, --app string   Delete a secret for a specific application: {actions|codespaces|dependabot}
  -e, --env string   Delete a secret for an environment
  -o, --org string   Delete a secret for an organization
  -u, --user	     Delete a secret for your user

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh secret list`

List secrets on one of the following levels:
- repository (default): available to GitHub Actions runs or Dependabot in a repository
- environment: available to GitHub Actions runs for a deployment environment in a repository
- organization: available to GitHub Actions runs, Dependabot, or Codespaces within an organization
- user: available to Codespaces for your user

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh secret list [flags]

ALIASES
  gh secret ls

FLAGS
  -a, --app string	  List secrets for a specific application: {actions|codespaces|dependabot}
  -e, --env string	  List secrets for an environment
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -o, --org string	  List secrets for an organization
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"
  -u, --user		  List a secret for your user

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  name, numSelectedRepos, selectedReposURL, updatedAt, visibility

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh secret set`

Set a value for a secret on one of the following levels:
- repository (default): available to GitHub Actions runs or Dependabot in a repository
- environment: available to GitHub Actions runs for a deployment environment in a repository
- organization: available to GitHub Actions runs, Dependabot, or Codespaces within an organization
- user: available to Codespaces for your user

Organization and user secrets can optionally be restricted to only be available to
specific repositories.

Secret values are locally encrypted before being sent to GitHub.


USAGE
  gh secret set <secret-name> [flags]

FLAGS
  -a, --app string	     Set the application for a secret: {actions|codespaces|dependabot}
  -b, --body string	     The value for the secret (reads from standard input if not specified)
  -e, --env environment	     Set deployment environment secret
  -f, --env-file file	     Load secret names and values from a dotenv-formatted file
      --no-repos-selected    No repositories can access the organization secret
      --no-store	     Print the encrypted, base64-encoded value instead of storing it on GitHub
  -o, --org organization     Set organization secret
  -r, --repos repositories   List of repositories that can access an organization or user secret
  -u, --user		     Set a secret for your user
  -v, --visibility string    Set visibility for an organization secret: {all|private|selected} (default "private")

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Paste secret value for the current repository in an interactive prompt
  $ gh secret set MYSECRET

  # Read secret value from an environment variable
  $ gh secret set MYSECRET --body "$ENV_VALUE"

  # Set secret for a specific remote repository
  $ gh secret set MYSECRET --repo origin/repo --body "$ENV_VALUE"

  # Read secret value from a file
  $ gh secret set MYSECRET < myfile.txt

  # Set secret for a deployment environment in the current repository
  $ gh secret set MYSECRET --env myenvironment

  # Set organization-level secret visible to both public and private repositories
  $ gh secret set MYSECRET --org myOrg --visibility all

  # Set organization-level secret visible to specific repositories
  $ gh secret set MYSECRET --org myOrg --repos repo1,repo2,repo3

  # Set organization-level secret visible to no repositories
  $ gh secret set MYSECRET --org myOrg --no-repos-selected

  # Set user-level secret for Codespaces
  $ gh secret set MYSECRET --user

  # Set repository-level secret for Dependabot
  $ gh secret set MYSECRET --app dependabot

  # Set multiple secrets imported from the ".env" file
  $ gh secret set -f .env

  # Set multiple secrets from stdin
  $ gh secret set -f - < myfile.txt

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ssh-key`

Manage SSH keys registered with your GitHub account.

USAGE
  gh ssh-key <command> [flags]

AVAILABLE COMMANDS
  add:		 Add an SSH key to your GitHub account
  delete:	 Delete an SSH key from your GitHub account
  list:		 Lists SSH keys in your GitHub account

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ssh-key add`

Add an SSH key to your GitHub account

USAGE
  gh ssh-key add [<key-file>] [flags]

FLAGS
  -t, --title string   Title for the new key
      --type string    Type of the ssh key: {authentication|signing} (default "authentication")

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ssh-key delete`

Delete an SSH key from your GitHub account

USAGE
  gh ssh-key delete <id> [flags]

FLAGS
  -y, --yes   Skip the confirmation prompt

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh ssh-key list`

Lists SSH keys in your GitHub account

USAGE
  gh ssh-key list [flags]

ALIASES
  gh ssh-key ls

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh status`

The status command prints information about your work on GitHub across all the repositories you're subscribed to, including:

- Assigned Issues
- Assigned Pull Requests
- Review Requests
- Mentions
- Repository Activity (new issues/pull requests, comments)


USAGE
  gh status [flags]

FLAGS
  -e, --exclude strings	  Comma separated list of repos to exclude in owner/name format
  -o, --org string	  Report status within an organization

INHERITED FLAGS
  --help   Show help for command

EXAMPLES
  $ gh status -e cli/cli -e cli/go-gh # Exclude multiple repositories
  $ gh status -o cli # Limit results to a single organization

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh variable`

Variables can be set at the repository, environment or organization level for use in
GitHub Actions or Dependabot. Run `gh help variable set` to learn how to get started.


USAGE
  gh variable <command> [flags]

AVAILABLE COMMANDS
  delete:	 Delete variables
  get:		 Get variables
  list:		 List variables
  set:		 Create or update variables

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh variable delete`

Delete a variable on one of the following levels:
- repository (default): available to GitHub Actions runs or Dependabot in a repository
- environment: available to GitHub Actions runs for a deployment environment in a repository
- organization: available to GitHub Actions runs or Dependabot within an organization


USAGE
  gh variable delete <variable-name> [flags]

ALIASES
  gh variable remove

FLAGS
  -e, --env string   Delete a variable for an environment
  -o, --org string   Delete a variable for an organization

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh variable get`

Get a variable on one of the following levels:
- repository (default): available to GitHub Actions runs or Dependabot in a repository
- environment: available to GitHub Actions runs for a deployment environment in a repository
- organization: available to GitHub Actions runs or Dependabot within an organization

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh variable get <variable-name> [flags]

FLAGS
  -e, --env string	  Get a variable for an environment
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -o, --org string	  Get a variable for an organization
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  createdAt, name, numSelectedRepos, selectedReposURL, updatedAt, value,
  visibility

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh variable list`

List variables on one of the following levels:
- repository (default): available to GitHub Actions runs or Dependabot in a repository
- environment: available to GitHub Actions runs for a deployment environment in a repository
- organization: available to GitHub Actions runs or Dependabot within an organization

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh variable list [flags]

ALIASES
  gh variable ls

FLAGS
  -e, --env string	  List variables for an environment
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -o, --org string	  List variables for an organization
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  createdAt, name, numSelectedRepos, selectedReposURL, updatedAt, value,
  visibility

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh variable set`

Set a value for a variable on one of the following levels:
- repository (default): available to GitHub Actions runs or Dependabot in a repository
- environment: available to GitHub Actions runs for a deployment environment in a repository
- organization: available to GitHub Actions runs or Dependabot within an organization

Organization variable can optionally be restricted to only be available to
specific repositories.


USAGE
  gh variable set <variable-name> [flags]

FLAGS
  -b, --body string	     The value for the variable (reads from standard input if not specified)
  -e, --env environment	     Set deployment environment variable
  -f, --env-file file	     Load variable names and values from a dotenv-formatted file
  -o, --org organization     Set organization variable
  -r, --repos repositories   List of repositories that can access an organization variable
  -v, --visibility string    Set visibility for an organization variable: {all|private|selected} (default "private")

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Add variable value for the current repository in an interactive prompt
  $ gh variable set MYVARIABLE

  # Read variable value from an environment variable
  $ gh variable set MYVARIABLE --body "$ENV_VALUE"

  # Read variable value from a file
  $ gh variable set MYVARIABLE < myfile.txt

  # Set variable for a deployment environment in the current repository
  $ gh variable set MYVARIABLE --env myenvironment

  # Set organization-level variable visible to both public and private repositories
  $ gh variable set MYVARIABLE --org myOrg --visibility all

  # Set organization-level variable visible to specific repositories
  $ gh variable set MYVARIABLE --org myOrg --repos repo1,repo2,repo3

  # Set multiple variables imported from the ".env" file
  $ gh variable set -f .env

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh workflow`

List, view, and run workflows in GitHub Actions.

USAGE
  gh workflow <command> [flags]

AVAILABLE COMMANDS
  disable:	 Disable a workflow
  enable:	 Enable a workflow
  list:		 List workflows
  run:		 Run a workflow by creating a workflow_dispatch event
  view:		 View the summary of a workflow

FLAGS
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

INHERITED FLAGS
  --help   Show help for command

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh workflow disable`

Disable a workflow, preventing it from running or showing up when listing workflows.

USAGE
  gh workflow disable [<workflow-id> | <workflow-name>] [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh workflow enable`

Enable a workflow, allowing it to be run and show up when listing workflows.

USAGE
  gh workflow enable [<workflow-id> | <workflow-name>] [flags]

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh workflow list`

List workflow files, hiding disabled workflows by default.

For more information about output formatting flags, see `gh help formatting`.

USAGE
  gh workflow list [flags]

ALIASES
  gh workflow ls

FLAGS
  -a, --all		  Include disabled workflows
  -q, --jq expression	  Filter JSON output using a jq expression
      --json fields	  Output JSON with the specified fields
  -L, --limit int	  Maximum number of workflows to fetch (default 50)
  -t, --template string	  Format JSON output using a Go template; see "gh help formatting"

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

JSON FIELDS
  id, name, path, state

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh workflow run`

Create a `workflow_dispatch` event for a given workflow.

This command will trigger GitHub Actions to run a given workflow file. The given workflow file must
support an `on.workflow_dispatch` trigger in order to be run in this way.

If the workflow file supports inputs, they can be specified in a few ways:

- Interactively
- Via `-f/--raw-field` or `-F/--field` flags
- As JSON, via standard input


USAGE
  gh workflow run [<workflow-id> | <workflow-name>] [flags]

FLAGS
  -F, --field key=value	      Add a string parameter in key=value format, respecting @ syntax (see "gh help api").
      --json		      Read workflow inputs as JSON via STDIN
  -f, --raw-field key=value   Add a string parameter in key=value format
  -r, --ref string	      Branch or tag name which contains the version of the workflow file you'd like to run

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Have gh prompt you for what workflow you'd like to run and interactively collect inputs
  $ gh workflow run

  # Run the workflow file 'triage.yml' at the remote's default branch
  $ gh workflow run triage.yml

  # Run the workflow file 'triage.yml' at a specified ref
  $ gh workflow run triage.yml --ref my-branch

  # Run the workflow file 'triage.yml' with command line inputs
  $ gh workflow run triage.yml -f name=scully -f greeting=hello

  # Run the workflow file 'triage.yml' with JSON via standard input
  $ echo '{"name":"scully", "greeting":"hello"}' | gh workflow run triage.yml --json

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

## `gh workflow view`

View the summary of a workflow

USAGE
  gh workflow view [<workflow-id> | <workflow-name> | <filename>] [flags]

FLAGS
  -r, --ref string   The branch or tag name which contains the version of the workflow file you'd like to view
  -w, --web	     Open workflow in the browser
  -y, --yaml	     View the workflow yaml file

INHERITED FLAGS
      --help			 Show help for command
  -R, --repo [HOST/]OWNER/REPO	 Select another repository using the [HOST/]OWNER/REPO format

EXAMPLES
  # Interactively select a workflow to view
  $ gh workflow view

  # View a specific workflow
  $ gh workflow view 0451

LEARN MORE
  Use `gh <command> <subcommand> --help` for more information about a command.
  Read the manual at https://cli.github.com/manual
  Learn about exit codes using `gh help exit-codes`
  Learn about accessibility experiences using `gh help accessibility`

---

