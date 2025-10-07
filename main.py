import subprocess

SCALE_STABLE_VERSION = "1.3.1"
SCALE_UNSTABLE_VERSION = "2025.03.24"

# Find the current git branch to choose stable or unstable variation.
g = subprocess.run(["git", "branch", "--show-current"], capture_output=True)
current_branch = g.stdout.decode("utf-8")[:-1]

print("Current branch: " + current_branch)

def define_env(env):
    env.variables["branch"] = current_branch
    env.variables["customer_specific_repo"] = "nonfree" in current_branch
    
    scale_pkgname = "scale"
    scale_version = SCALE_UNSTABLE_VERSION if "unstable" in current_branch else SCALE_STABLE_VERSION

    repo_subdomain = "pkgs"
    if "nonfree" in current_branch:
        repo_subdomain = "nonfree-" + repo_subdomain
    else:
        scale_pkgname += "-free"

    if "unstable" in current_branch:
        scale_pkgname += "-unstable"

    env.variables["scale_pkgname"] = scale_pkgname
    env.variables["scale_version"] = scale_version
    env.variables["repo_subdomain"] = repo_subdomain

    @env.macro
    def checksum(url):
        g = subprocess.run(["curl", url + ".sha512"], capture_output=True)
        return g.stdout.decode("utf-8").split(" ")[0]
